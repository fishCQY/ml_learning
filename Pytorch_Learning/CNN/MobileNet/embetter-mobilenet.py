  # @file    embetter-mobilenet.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/05 09:01:26
  # @version 1.0
  # @brief 

import os
import pandas as pd

for name in ("train", "test"):  # 同时处理训练集和测试集
    # 读取原始CSV文件（假设文件名为train.csv和test.csv）
    df = pd.read_csv(f"Pytorch_Learning/datasets/mnist-pngs/{name}.csv")  
    
    # 修改filepath列：在所有路径前添加"mnist-pngs/"前缀
    df["filepath"] = df["filepath"].apply(lambda x: "mnist-pngs/" + x)
    
    # 随机打乱数据顺序：
    # sample(frac=1)表示100%采样（即全部数据但顺序随机）
    # random_state=123确保每次打乱结果相同（可复现）
    # reset_index(drop=True)重置索引并丢弃旧索引
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # 保存处理后的CSV到mnist-pngs目录
    # 文件名为train_shuffled.csv和test_shuffled.csv
    # index=None表示不保存行索引
    df.to_csv(f"Pytorch_Learning/datasets/mnist-pngs/{name}_shuffled.csv", index=None)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from tqdm.notebook import tqdm 

# pip install "embetter[vision]"
from embetter.vision import ImageLoader, TimmEncoder

embed = make_pipeline(
    ImageLoader(),
    TimmEncoder(name="mobilenetv3_large_100")
)


