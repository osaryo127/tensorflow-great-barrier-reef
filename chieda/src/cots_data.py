import os.path as osp
import shutil
import ast
import numpy as np
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm

# ROOT_PATH = "/kaggle/input/tensorflow-great-barrier-reef/"
ROOT_PATH = "../"


def process_data(data):
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


if __name__ == "__main__":
    df = pd.read_csv(osp.join(ROOT_PATH, "train.csv"))

    # df.annotations = df.annotations.apply(ast.literal_eval)
    # df = df[df.annotations.apply(len) > 0].reset_index(drop=True)
    # df_tr, df_va = model_selection.train_test_split(
    #     df, test_size=0.25, random_state=42, shuffle=True
    # )
    # df_tr = df_tr.reset_index(drop=True)
    # df_va = df_va.reset_index(drop=True)
    # process_data(df_tr, data_type="train")
    # process_data(df_va, data_type="validation")
