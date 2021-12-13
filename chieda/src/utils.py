import os.path as osp
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from albumentations.augmentations import transforms
from sklearn import model_selection
import torch
import albumentations as A

import dataset as ds
from run import ROOT_PATH
from yolov5.cots_data import OUTPUT_PATH

ROOT = Path("../")


def add_fold(df):
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
    df_split = (
        df.groupby("subsequence_id").agg({"has_annotations": "max", "video_frame": "count"}).astype(int).reset_index()
    )
    n_splits = 5
    kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold_id, (_, val_idx) in enumerate(kf.split(df_split["subsequence_id"], y=df_split["has_annotations"])):
        subseq_val_idx = df_split["subsequence_id"].iloc[val_idx]
        df.loc[df["subsequence_id"].isin(subseq_val_idx), "fold"] = fold_id
        print(f"fold {fold_id} : {subseq_val_idx.values}")

    df["fold"] = df["fold"].astype(int)
    # [TODO] write how subsequence is assigned to folds

    return df


def process_data(data, data_type="train"):
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
            osp.join(ROOT_PATH, f"labels/{data_type}/{image_name}.txt"), data_, fmt=["%d", "%f", "%f", "%f", "%f"],
        )


def get_augmentation(opt):
    train_transform = []
    return A.OneOf(train_transform)


def get_dataloader(opt):
    classes = ["cots"]
    color_mean = (104, 117, 123)  # BGR
    img_tr, anno_tr, img_va, anno_va = ds.make_datapath(ROOT)

    imgsz = 720
    dataset_tr = ds.CotsDataset(
        img_tr,
        anno_tr,
        phase="train",
        transform=ds.CotsTransform(imgsz, color_mean),
        transform_anno=ds.AnnoList(classes),
    )
    dataset_va = ds.CotsDataset(
        img_va, anno_va, phase="val", transform=ds.CotsTransform(imgsz, color_mean), transform_anno=ds.AnnoList(classes)
    )
    batch_size = opt.batchsize
    dataloader_tr = torch.utils.DataLoader(dataset_tr, batch_size=batch_size, suffle=True, collate_fn=anno_collater)
    dataloader_va = torch.utils.DataLoader(dataset_va, batch_size=batch_size, suffle=False, collate_fn=anno_collater)
    return dataloader_tr, dataloader_va
