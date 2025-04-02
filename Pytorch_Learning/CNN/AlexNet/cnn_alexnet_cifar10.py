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

import matplotlib.pylot as plt
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True

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

random_seed =1
learning_rate = 0.0001
batch_size = 256
num_epochs =40

num_classes =10

device = "cuda:0"
set_all_seeds(random_seed)

import sys
sys.path.insert(0,"../Pytorch_Learning/utils/helper_evaluate.py")
from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss

