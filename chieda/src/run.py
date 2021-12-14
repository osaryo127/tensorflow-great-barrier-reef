import os.path as osp
from pathlib import Path
import yaml
import pandas as pd

from utils import add_fold, label2txtfile
from dataset import get_dataset, get_dataloader

# ROOT_PATH = "/kaggle/input/tensorflow-great-barrier-reef/"
ROOT_PATH = Path("../")


def run(opt=None):
    """Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')"""
    with open("../config/experiment.yaml") as f:
        opt = yaml.safe_load(f)
    df = pd.read_csv(osp.join(ROOT_PATH, "train.csv"))
    # 交差検証用のFOLDを付与
    df = add_fold(df, opt)
    # データセット取得
    # for fold in range(max(df.fold)):
    ds_tr, ds_va = get_dataset(df, 0, opt)
    # データローダー取得
    dl_tr, dl_va = get_dataloader(ds_tr, ds_va, opt)
    # # モデル取得
    # net = get_model(opt)
    # # デバイス取得
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using", device)
    # # ロス関数取得
    # criterion = get_criterion(opt)
    # # 最適化関数
    # optimizer = get_optimizer(opt)
    # # 訓練
    # train(opt)
    # # 推論
    # inference(opt)
    # # 結果の保存
    # write_log(opt)


if __name__ == "__main__":
    run()
