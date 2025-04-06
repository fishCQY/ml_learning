  # @file    gan.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/06 15:10:35
  # @version 1.0
  # @brief 生成对抗网络

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
generator_learning_rate = 0.001
discriminator_learning_rate = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 128
LATENT_DIM = 75 #潜在空间维度
IMG_SHAPE = (1, 28, 28)
IMG_SIZE = 1
for x in IMG_SHAPE: #乘以每个维度的大小，得到图片的总大小
    IMG_SIZE *= x



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
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

##########################
### MODEL
##########################
class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        # 生成器（Generator）
        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),  # 输入：噪声向量（潜在空间）
            nn.LeakyReLU(inplace=True),  # 激活函数
            nn.Dropout(p=0.5),           # 正则化防止过拟合
            nn.Linear(128, IMG_SIZE),    # 输出：生成图像（展平后的向量）
            nn.Tanh()                    # 将输出限制到[-1, 1]范围
        )
        
        # 判别器（Discriminator）
        self.discriminator = nn.Sequential(
            nn.Linear(IMG_SIZE, 128),    # 输入：展平的图像（真实或生成）
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),           # 输出：判别概率（0假/1真）
            nn.Sigmoid()                 # 转换为概率值[0,1]
        )

    def generator_forward(self, z):
        """生成器前向传播：输入噪声z，生成图像"""
        return self.generator(z)

    def discriminator_forward(self, img):
        """判别器前向传播：输入图像，输出真伪概率"""
        return self.discriminator(img).view(-1)  # 展平为1D张量

torch.manual_seed(random_seed)

model = GAN()
model = model.to(device)

optim_gener = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_learning_rate)

start_time = time.time()    

# 初始化损失记录列表
discr_costs = []  # 判别器损失记录
gener_costs = []  # 生成器损失记录

# 开始训练循环
for epoch in range(NUM_EPOCHS):
    model = model.train()  # 设置模型为训练模式
    
    # 遍历训练数据
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        # 数据预处理
        features = (features - 0.5)*2.  # 将图像像素值从[0,1]归一化到[-1,1]
        features = features.view(-1, IMG_SIZE).to(device)  # 展平图像并转移到设备转换成2D张量，batchsize,imgsize
        targets = targets.to(device)  # 转移标签到设备

        # 创建真实和假标签
        valid = torch.ones(targets.size(0)).float().to(device)  # 真实样本标签为1
        fake = torch.zeros(targets.size(0)).float().to(device)  # 生成样本标签为0

        # --------------------------
        # 训练生成器
        # --------------------------
        
        # 生成新图像
        z = torch.zeros((targets.size(0), LATENT_DIM)).uniform_(-1.0, 1.0).to(device)  # 生成随机噪声
        generated_features = model.generator_forward(z)  # 生成假图像
        
        # 计算生成器损失（欺骗判别器）
        discr_pred = model.discriminator_forward(generated_features)  # 判别器对生成图像的判断
        gener_loss = F.binary_cross_entropy(discr_pred, valid)  # 希望判别器将生成图像判断为真
        
        # 反向传播和优化
        optim_gener.zero_grad()  # 清空生成器梯度
        gener_loss.backward()    # 反向传播
        optim_gener.step()       # 更新生成器参数

        # --------------------------
        # 训练判别器
        # --------------------------        
        
        # 计算真实图像损失
        discr_pred_real = model.discriminator_forward(features.view(-1, IMG_SIZE))
        real_loss = F.binary_cross_entropy(discr_pred_real, valid)  # 希望判别器正确识别真实图像
        
        # 计算生成图像损失（使用detach()防止生成器更新）
        discr_pred_fake = model.discriminator_forward(generated_features.detach())
        fake_loss = F.binary_cross_entropy(discr_pred_fake, fake)  # 希望判别器正确识别生成图像
        
        # 总判别器损失（真实和生成图像损失的平均）
        discr_loss = 0.5*(real_loss + fake_loss)

        # 反向传播和优化
        optim_discr.zero_grad()  # 清空判别器梯度
        discr_loss.backward()    # 反向传播
        optim_discr.step()       # 更新判别器参数
        
        # 记录损失
        discr_costs.append(discr_loss.item())  # 记录判别器损失
        gener_costs.append(gener_loss.item())  # 记录生成器损失
        
        # 每100个batch打印一次训练信息
        if not batch_idx % 100:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), gener_loss, discr_loss))

    # 打印每个epoch耗时
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
# 打印总训练时间
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

import matplotlib.pyplot as plt
ax1 = plt.subplot(1, 1, 1)
ax1.plot(range(len(gener_costs)), gener_costs, label='Generator loss')
ax1.plot(range(len(discr_costs)), discr_costs, label='Discriminator loss')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.legend()

###################
# 设置第二个x轴（显示epoch数）
ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
newlabel = list(range(NUM_EPOCHS+1))  # 生成epoch标签 [0,1,2,...,NUM_EPOCHS]
iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
newpos = [e*iter_per_epoch for e in newlabel]  # 计算每个epoch对应的迭代位置

# 设置刻度标签和位置（每10个epoch显示一次）
ax2.set_xticklabels(newlabel[::10])  
ax2.set_xticks(newpos[::10])

# 调整第二个x轴的位置和样式
ax2.xaxis.set_ticks_position('bottom')  # 刻度线在下方
ax2.xaxis.set_label_position('bottom')  # 标签在下方
ax2.spines['bottom'].set_position(('outward', 45))  # 将第二个x轴下移45点
ax2.set_xlabel('Epochs')  # 设置轴标签
ax2.set_xlim(ax1.get_xlim())  # 与主x轴范围一致
###################

plt.show()  # 显示图像

##########################
### 生成图像可视化
##########################

model.eval()  # 设置模型为评估模式
# 生成新图像
z = torch.zeros((5, LATENT_DIM)).uniform_(-1.0, 1.0).to(device)  # 生成5个随机噪声向量
generated_features = model.generator_forward(z)  # 通过生成器生成图像特征
imgs = generated_features.view(-1, 28, 28)  # 调整形状为(5,28,28)的MNIST图像格式

# 创建1行5列的子图
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 2.5))

# 在每个子图中显示生成的图像
for i, ax in enumerate(axes):
    # 将张量转移到CPU并分离计算图，使用灰度色图显示
    axes[i].imshow(imgs[i].to(torch.device('cpu')).detach(), cmap='binary')