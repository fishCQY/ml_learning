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
    df["filepath"] = df["filepath"].apply(lambda x: "Pytorch_Learning/datasets/mnist-pngs/" + x)
    
    # 随机打乱数据顺序：
    # sample(frac=1)表示100%采样（即全部数据但顺序随机）
    # random_state=123确保每次打乱结果相同（可复现）
    # reset_index(drop=True)重置索引并丢弃旧索引
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # 保存处理后的CSV到mnist-pngs目录
    # 文件名为train_shuffled.csv和test_shuffled.csv
    # index=None表示不保存行索引
    df.to_csv(f"Pytorch_Learning/datasets/mnist-pngs/{name}_shuffled.csv", index=None)

# ​​make_pipeline​​:
# 用于将多个数据处理步骤（如特征提取、分类器）串联成一个端到端的机器学习管道。

# ​​示例​​：make_pipeline(步骤1, 步骤2, 步骤3) 会自动按顺序执行这些步骤。
# ​​SGDClassifier​​:
# 一个基于随机梯度下降（Stochastic Gradient Descent）的线性分类器，适用于大规模数据集和高维特征。
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from tqdm.notebook import tqdm   # 进度条显示

# pip install "embetter[vision]"Embetter 图像处理组件​
from embetter.vision import ImageLoader, TimmEncoder

# 创建特征提取流水线
embed = make_pipeline(
    ImageLoader(),  # 图像加载器：从文件路径读取图像
    TimmEncoder(name="mobilenetv3_large_100")  # 使用MobileNetV3提取特征
)

# 初始化SGD分类器（逻辑回归）
model = SGDClassifier(
    loss="log_loss",  # 使用对数损失（逻辑回归）
    n_jobs=-1,        # 使用所有CPU核心
    shuffle=True      # 每次迭代前打乱数据
)

# 配置参数
chunksize = 1000      # 每次处理1000个样本
train_labels, train_predict = [], []  # 初始化记录容器

# 分块读取并处理训练数据
for df in tqdm(pd.read_csv("Pytorch_Learning/datasets/mnist-pngs/train_shuffled.csv", 
                          chunksize=chunksize,  # 每次读取1000行
                          iterator=True), 
              total=60):  # 预计总迭代次数
    
    # 特征提取：将图像路径转换为MobileNetV3特征向量
    embedded = embed.transform(df["filepath"])
    
    # 增量训练：使用部分数据更新模型参数
    model.partial_fit(embedded,          # 特征向量
                     df["label"],        # 标签
                     classes=list(range(10)))  # 指定类别数(0-9)

# 初始化两个空列表，用于存储：
# train_labels - 真实标签
# train_predict - 模型预测结果
train_labels, train_predict = [], []

# 分块读取训练集CSV文件（每次处理1000个样本，共60次迭代）
for df in tqdm(pd.read_csv("Pytorch_Learning/datasets/mnist-pngs/train.csv",
                          chunksize=chunksize,  # 每块1000样本
                          iterator=True), 
              total=60):  # 总进度60次
    
    # 修正图像路径：在所有路径前添加"mnist-pngs/"前缀
    df["filepath"] = df["filepath"].apply(lambda x: "Pytorch_Learning/datasets/mnist-pngs/" + x)
    
    # 特征提取：使用MobileNetV3转换图像为特征向量
    embedded = embed.transform(df["filepath"])
    
    # 模型预测：对当前数据块进行预测
    train_predict.extend(model.predict(embedded))  # 收集预测结果
    
    # 收集真实标签（转换为Python列表后扩展）
    train_labels.extend(list(df["label"].values))

test_labels, test_predict = [], []

# 分块读取测试集CSV文件（每次处理1000个样本，共10次迭代）
for df in tqdm(pd.read_csv("Pytorch_Learning/datasets/mnist-pngs/test_shuffled.csv",
                          chunksize=chunksize,  # 每块1000样本
                          iterator=True),
              total=10):  # 总进度10次（约1万测试样本）
    
    # 特征提取：使用MobileNetV3转换图像为特征向量
    embedded = embed.transform(df["filepath"])
    
    # 模型预测：对当前数据块进行预测
    test_predict.extend(model.predict(embedded))  # 收集预测结果
    
    # 收集真实标签（转换为Python列表后扩展）
    test_labels.extend(list(df["label"].values))

from sklearn.metrics import accuracy_score

print(f"Train accuracy: {accuracy_score(train_labels, train_predict):.2f}")
print(f"Test accuracy: {accuracy_score(test_labels, test_predict):.2f}")