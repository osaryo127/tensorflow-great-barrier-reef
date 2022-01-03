import os.path as osp
from pathlib import Path
import ast
import random
import numpy as np
import pandas as pd
import imagesize
from tqdm import tqdm

from sklearn import model_selection
import cv2
import albumentations as A
import torch

import dataset as ds

ROOT = Path("../")


def add_fold(df: pd.DataFrame, n_folds: int = 5, seed: int = 42, only_positive: bool = True, show_fold_info: bool = False) -> pd.DataFrame:
    """train.csvに交差検証用のFOLDを付与する"""
    # アノテーション数
    df["annotations"] = df["annotations"].apply(ast.literal_eval)
    df["n_annotations"] = df["annotations"].str.len()
    df["has_annotations"] = df["annotations"].str.len() > 0
    df["doesnt_have_annotations"] = df["annotations"].str.len() == 0

    # 物体の有無によるシーケンスの分割
    df["start_cut_here"] = (
        df["has_annotations"] & df["doesnt_have_annotations"].shift(1) & df["doesnt_have_annotations"].shift(2)
    )
    df["end_cut_here"] = df["doesnt_have_annotations"] & df["has_annotations"].shift(1) & df["has_annotations"].shift(2)
    df["sequence_change"] = df["sequence"] != df["sequence"].shift(1)
    df["last_row"] = df.index == len(df) - 1
    df["cut_here"] = df["start_cut_here"] | df["end_cut_here"] | df["sequence_change"] | df["last_row"]
    start_idx = 0
    for subsequence_id, end_idx in enumerate(df[df["cut_here"]].index):
        df.loc[start_idx:end_idx, "subsequence_id"] = subsequence_id
        start_idx = end_idx
    df["subsequence_id"] = df["subsequence_id"].astype(int)
    drop_cols = ["start_cut_here", "end_cut_here", "sequence_change", "last_row", "cut_here", "doesnt_have_annotations"]
    df = df.drop(drop_cols, axis=1)

    # 物体の有無による階層化を用いたsubsequenceのfold振り分け
    df_split = (df.groupby("subsequence_id").agg({"has_annotations": "max", "video_frame": "count"}).astype(int).reset_index())
    if only_positive:
        df = df[df.has_annotations].reset_index(drop=True)
        kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_id, (_, val_idx) in enumerate(kf.split(df_split["subsequence_id"])):
            subseq_val_idx = df_split["subsequence_id"].iloc[val_idx]
            df.loc[df["subsequence_id"].isin(subseq_val_idx), "fold"] = fold_id
            if show_fold_info:
                print(f"fold {fold_id} : {subseq_val_idx.values}")
    else:
        kf = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_id, (_, val_idx) in enumerate(kf.split(df_split["subsequence_id"], y=df_split["has_annotations"])):
            subseq_val_idx = df_split["subsequence_id"].iloc[val_idx]
            df.loc[df["subsequence_id"].isin(subseq_val_idx), "fold"] = fold_id
            if show_fold_info:
                print(f"fold {fold_id} : {subseq_val_idx.values}")

    df["fold"] = df["fold"].astype(int)

    return df


def label2txtfile(data, data_type="train"):
    """train.csvから、
    """
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row["image_id"]
        annos = row["annotations"]
        data_ = []
        for anno in annos:
            x_center = anno["x"] + anno["width"] / 2
            y_center = anno["y"] + anno["height"] / 2
            x_center /= 1280
            y_center /= 720
            w = anno["width"] / 1280
            h = anno["height"] / 720
            data_.append([0, x_center, y_center, w, h])
        data_ = np.array(data_)
        np.savetxt(
            osp.join(ROOT, f"labels/{data_type}/{image_name}.txt"), data_, fmt=["%d", "%f", "%f", "%f", "%f"],
        )


def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
    この変化に対応したDataLoaderを作成するために、
    カスタイマイズした、collate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像imgです
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] はアノテーションgtです

    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリストです。
    # リストのサイズはミニバッチサイズです。
    # リストtargetsの要素は [n, 5] となっています。
    # nは画像ごとに異なり、画像内にある物体の数となります。
    # 5は [xmin, ymin, xmax, ymax, class_index] です

    return imgs, targets


def get_augmentation(opt):
    train_transform = []
    return A.OneOf(train_transform)
