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