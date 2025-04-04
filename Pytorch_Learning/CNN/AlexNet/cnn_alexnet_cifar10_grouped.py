# @file    cnn_alexnet_cifar10.py
# @author  cqy 3049623863@qq.com
# @date    2025/04/02 19:39:33
# @version 1.0
# @brief

import os
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)


random_seed = 1
learning_rate = 0.0001
batch_size = 256
num_epochs = 40

num_classes = 10

device = "cuda:0"
set_all_seeds(random_seed)

import sys
import os

# 获取当前脚本的绝对路径（示例值）
script_path = os.path.abspath(__file__)
print("[1] 当前脚本路径:", script_path)
# 计算正确的项目根目录（Pytorch_Learning）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
# 分解步骤：
# 1. os.path.dirname(script_path) → D:\ml_learning\Pytorch_Learning\CNN\AlexNet
# 2. 再调用一次 → D:\ml_learning\Pytorch_Learning\CNN
# 3. 再调用一次 → D:\ml_learning\Pytorch_Learning

# 构建工具目录路径
utils_dir = os.path.join(project_root, "utils")
utils_dir = os.path.normpath(utils_dir)
print("计算后的工具目录:", utils_dir)  # 输出: D:\ml_learning\Pytorch_Learning\utils

# 验证路径是否存在
if not os.path.exists(utils_dir):
    raise FileNotFoundError(f"❌ 目录不存在: {utils_dir}")
print("✅ 目录存在性验证通过")

# 添加到 sys.path
sys.path.insert(0, utils_dir)

# 导入模块
from helper_evaluate import compute_accuracy
from helper_data import get_dataloaders_cifar10
from helper_train import train_classifier_simple_v1

# CIFAR-10 数据集详解
# 1. ​基本概述
# ​用途：图像分类任务（监督学习）。
# ​规模：包含 ​60,000 张彩色图像，分为 ​10 个类别，每类 ​6,000 张。
# ​训练集：50,000 张（每类 5,000 张）。
# ​测试集：10,000 张（每类 1,000 张）。
# ​图像属性：
# ​分辨率：32×32 像素。
# ​通道：RGB 彩色（3 通道）。
# ​类别：飞机（airplane）、汽车（automobile）、鸟（bird）、猫（cat）、鹿（deer）、狗（dog）、青蛙（frog）、马（horse）、船（ship）、卡车（truck）。

set_all_seeds(random_seed)
# ​Resize(70,70)：
# 统一图像大小，确保后续裁剪操作的有效性（尤其当原始图像尺寸不一致时）。
# ​RandomCrop(64,64)：
# ​随机位置裁剪，增加数据多样性，防止模型过拟合（每次训练看到不同局部）。
# ​CenterCrop(64,64)：
# ​中心位置裁剪，保证评估时图像处理方式确定性（结果可复现）。
# ​ToTensor()：
# 将PIL图像或NumPy数组转换为 [C, H, W] 格式的Tensor，并自动归一化到 [0, 1]
train_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                       transforms.RandomCrop((64, 64)),
                                       transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                      transforms.CenterCrop((64, 64)),
                                      transforms.ToTensor()])
train_loader, valid_loader, test_loader = get_dataloaders_cifar10(batch_size=batch_size,
                                                                  num_workers=0,
                                                                  train_transforms=train_transforms,
                                                                  test_transforms=test_transforms,
                                                                  validation_fraction=0.1)

# checking the dataset
print('Training Set:\n')
for images, labels in train_loader:
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    break

# checking the dataset
print('\nValidation Set:')
for images, labels in valid_loader:
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break

# checking the dataset
print('\nTesting Set:')
for images, labels in train_loader:
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # 特征提取网络（带分组卷积的改进版）
        self.features = nn.Sequential(
            # 第1层（标准卷积层）
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 输入通道3 → 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出尺寸64@7x7

            # 第2层（分组卷积）groups=2
            nn.Conv2d(64, 192, kernel_size=5, padding=2, groups=2),  # 输入分2组 → 每组输出96通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出192@3x3

            # 后续连续3层分组卷积（groups=2）
            nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2),  # 每组输出192通道
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),  # 每组输出128通道
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=2),  # 每组输出128通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 最终特征图256x1x1
        )
        
        # 自适应平均池化：将任意尺寸特征图统一到 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 输出: [256, 6, 6]

        # 分类网络（全连接部分）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 50% dropout 防止过拟合
            nn.Linear(256 * 6 * 6, 4096),  # 输入特征展开为 9216 维
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # 全连接层
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # 最终分类层
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = self.avgpool(x)   # 自适应平均池化
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits


torch.manual_seed(random_seed)
model = AlexNet(num_classes=num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

log_dict = train_classifier_simple_v1(
    num_epochs=num_epochs,
    model=model,
    optimizer=optimizer,
    device=device,
    train_loader=train_loader,
    valid_loader=valid_loader,
    logging_interval=50
)

import matplotlib.pyplot as plt

loss_list = log_dict['train_loss_per_batch']
# 训练损失可视化代码解析
plt.plot(loss_list, label='Minibatch loss')  # 原始小批次损失曲线
#np.convolve 是 NumPy 中用于计算一维卷积的函数
plt.plot(np.convolve(
    loss_list,  # 原始损失序列（每个batch的损失值）
    np.ones(200) / 200,  # 200点的平均滤波器（每个点代表前200个batch的移动平均）
    mode='valid'          # 有效卷积模式（仅计算完全重叠的区域，输出长度为 len(loss_list)-199）
), label='Running average')  # 添加滑动平均曲线

plt.ylabel('Cross Entropy')  # Y轴标签：交叉熵损失值
plt.xlabel('Iteration')       # X轴标签：迭代次数（1个iteration=1个batch训练）
plt.legend()                  # 显示图例
plt.show()                    # 渲染图像窗口

# 代码功能说明：
# 1. 蓝线（Minibatch loss）：原始训练损失曲线，反映每个batch的即时波动
# 2. 橙线（Running average）：200个batch的移动平均线，用于观察损失的整体下降趋势
# 3. 有效卷积模式：丢弃前199个不完整平均的数据点，保证曲线平滑度与统计显著性
# 4. 横纵坐标标注：明确显示度量的指标（交叉熵损失）和训练进度（迭代次数）

# 应用场景：监控模型训练过程，判断是否出现：
# - 欠拟合（损失居高不下）
# - 过拟合（训练损失下降但验证损失上升）
# - 收敛情况（损失曲线趋于平稳）

plt.plot(np.arange(1, num_epochs + 1), log_dict['train_acc_per_epoch'], label='Training')
plt.plot(np.arange(1, num_epochs + 1), log_dict['valid_acc_per_epoch'], label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

with torch.set_grad_enabled(False):
    train_acc = compute_accuracy(model=model,
                                 data_loader=test_loader,
                                 device=device)
    test_acc = compute_accuracy(model=model,
                                data_loader=test_loader,
                                device=device)
    valid_acc = compute_accuracy(model=model,
                                 data_loader=valid_loader,
                                 device=device)
    print(f'Train ACC:{valid_acc:.2f}%')
    print(f'Validation ACC:{valid_acc:.2f}%')
    print(f'Test ACC:{test_acc:.2f}%')
