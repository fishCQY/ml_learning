  # @file    autoencoder-mnist.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/05 16:08:14
  # @version 1.0
  # @brief 

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 5
batch_size = 256

# Architecture
num_features = 784
num_hidden_1 = 32


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='/root/autodl-fs/ml_learning/Pytorch_Learning/datasets', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='/root/autodl-fs/ml_learning/Pytorch_Learning/datasets', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

##########################
### MODEL
##########################

class Autoencoder(torch.nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        ### ENCODER
        # 编码器：将高维输入压缩到低维潜在空间
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # He初始化适配LeakyReLU（虽然此处标准差设为0.1，通常应使用1/√n）
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()  # 偏置归零初始化
        
        ### DECODER
        # 解码器：从潜在空间重建原始数据
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()

    def forward(self, x):
        # 编码阶段：降维 + 非线性激活,使用leaky relu
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)  # 负区间梯度保留（防止神经元死亡）
        
        # 解码阶段：升维 + 值域压缩,使用sigmoid
        logits = self.linear_2(encoded)
        decoded = torch.sigmoid(logits)  # 映射到[0,1]匹配像素值范围
        
        return decoded

# 初始化与设备配置
torch.manual_seed(random_seed)  # 固定随机性
model = Autoencoder(num_features=num_features)  # 假设num_features=784（28x28）
model.to(device)  # GPU加速
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 自适应学习率优化器

##########################
### TRAINING
##########################

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        # 数据预处理：展平图像（MNIST 28x28→784）
        features = features.view(-1, 28 * 28).to(device)
        
        # 前向传播
        decoded = model(features)  # 获取重建图像
        
        # 损失计算：二元交叉熵（适用于二值化像素）
        cost = F.binary_cross_entropy(decoded, features)
        
        # 反向传播与参数更新
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # 训练日志（每50个batch输出一次）
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                  %(epoch+1, num_epochs, batch_idx, 
                    len(train_loader), cost))

##########################
### VISUALIZATION
##########################
import matplotlib.pyplot as plt
# 可视化对比原始图像与重建结果
n_images = 15  # 显示15组对比
fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(20, 2.5))
orig_images = features[:n_images]  # 原始图像批次
decoded_images = decoded[:n_images]  # 重建图像批次

# 绘制图像（CPU处理）
for i in range(n_images):
    for ax, img in zip(axes, [orig_images, decoded_images]):
        curr_img = img[i].detach().cpu()  # 脱离计算图+转CPU
        ax[i].imshow(curr_img.view(28, 28), cmap='binary')  # 28x28灰度显示