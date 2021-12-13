import os.path as osp
from pathlib import Path
import yaml
import pandas as pd
import torch

from utils import add_fold, process_data

# ROOT_PATH = "/kaggle/input/tensorflow-great-barrier-reef/"
ROOT_PATH = Path("../")


def run(opt=None):
    """Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')"""
    with open("../config/experiment.yaml") as f:
        opt = yaml.safe_load(f)
    df = pd.read_csv(osp.join(ROOT_PATH, "train.csv"))
    df = add_fold(df)
    process_data(df)
    dl = get_dataloader(opt)
    # net = get_model(opt)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using", device)
    # criterion = get_criterion(opt)
    # optimizer = get_optimizer(opt)
    # train(opt)
    # inference(opt)
    write_log(opt)


if __name__ == "__main__":
    run()
