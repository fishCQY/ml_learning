  # @file    helper_utils.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/04 21:33:22
  # @version 1.0
  # @brief 

import os
import random
import numpy as np
import torch


def set_all_seeds(seed):
    """设置所有随机种子以保证实验可重复性
    
    参数:
        seed (int): 随机种子值，范围建议0-4294967295
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)  # 设置PyTorch Lightning全局种子
    random.seed(seed)        # 设置Python内置随机模块种子
    np.random.seed(seed)      # 设置NumPy随机种子
    torch.manual_seed(seed)   # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有CUDA设备的随机种子


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)