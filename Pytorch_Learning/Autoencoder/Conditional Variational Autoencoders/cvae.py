  # @file    cvae.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/06 11:31:42
  # @version 1.0
  # @brief 条件变分自编码器，使得生成受特定条件的数据，在模型中引入条件信息


import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 0
learning_rate = 0.001
num_epochs = 50
batch_size = 128

# Architecture
num_classes = 10
num_features = 784
num_latent = 50

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


def to_onehot(labels, num_classes, device):

    labels_onehot = torch.zeros(labels.size()[0], num_classes).to(device)
# labels.view(-1, 1) - 将原始标签向量变形为列向量（形状从[batch_size]变为[batch_size, 1]）
# scatter_(dim, index, value) - 是PyTorch的原地散射操作，参数说明：
# dim=1：表示按行操作（在每行的指定列位置填充值）
# index=labels.view(-1,1)：指定每行要填充的列位置（即类别索引）
# value=1：在指定位置填充1 ，scatter_（带下划线）表示原地修改张量

    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot


class ConditionalVariationalAutoencoder(torch.nn.Module):

    def __init__(self, num_features, num_latent, num_classes):
        """条件变分自编码器(CVAE)的初始化
        
        Args:
            num_features: 输入特征维度(784 for MNIST)
            num_latent: 潜在空间维度
            num_classes: 类别数量(10 for MNIST)
        """
        super(ConditionalVariationalAutoencoder, self).__init__()
        
        self.num_classes = num_classes  # 存储类别数
        
        ###############
        # 编码器部分
        ##############
        
        # 第一卷积层: 输入通道=1(图像)+10(条件), 输出16通道
        self.enc_conv_1 = torch.nn.Conv2d(in_channels=1+self.num_classes,
                                          out_channels=16,
                                          kernel_size=(6, 6),
                                          stride=(2, 2),  # 下采样
                                          padding=0)  # 无填充

        # 第二卷积层: 16->32通道
        self.enc_conv_2 = torch.nn.Conv2d(in_channels=16,
                                          out_channels=32,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),  # 下采样
                                          padding=0)                 
        
        # 第三卷积层: 32->64通道
        self.enc_conv_3 = torch.nn.Conv2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),  # 下采样
                                          padding=0)                     
        
        # 潜在空间均值估计层
        self.z_mean = torch.nn.Linear(64*2*2, num_latent)
        # 潜在空间对数方差估计层(使用对数方差确保数值稳定性)
        self.z_log_var = torch.nn.Linear(64*2*2, num_latent)
        
        
        ###############
        # 解码器部分
        ##############
        
        # 全连接层: 潜在向量+条件 -> 64*2*2
        self.dec_linear_1 = torch.nn.Linear(num_latent+self.num_classes, 64*2*2)
               
        # 第一转置卷积层: 64->32通道
        self.dec_deconv_1 = torch.nn.ConvTranspose2d(in_channels=64,
                                                     out_channels=32,
                                                     kernel_size=(2, 2),
                                                     stride=(2, 2),  # 上采样
                                                     padding=0)
                                 
        # 第二转置卷积层: 32->16通道
        self.dec_deconv_2 = torch.nn.ConvTranspose2d(in_channels=32,
                                                     out_channels=16,
                                                     kernel_size=(4, 4),
                                                     stride=(3, 3),  # 上采样
                                                     padding=1)
        
        # 第三转置卷积层: 16->11通道(11=1图像+10条件)
        self.dec_deconv_3 = torch.nn.ConvTranspose2d(in_channels=16,
                                                     out_channels=11,
                                                     kernel_size=(6, 6),
                                                     stride=(3, 3),  # 上采样
                                                     padding=4)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self, features, targets):
        """编码器前向传播过程，将输入图像和条件标签编码为潜在空间分布参数
        
        Args:
            features: 输入图像张量，形状为(batch_size, 1, height, width)
            targets: 类别标签张量，形状为(batch_size,)
            
        Returns:
            z_mean: 潜在空间均值，形状为(batch_size, num_latent)
            z_log_var: 潜在空间对数方差，形状同z_mean
            encoded: 重参数化后的潜在变量，形状同z_mean
        """
        ### 条件信息处理 ###
        # 将标签转为one-hot编码 (batch_size, num_classes)
        onehot_targets = to_onehot(targets, self.num_classes, device)
        # 调整形状为(batch_size, num_classes, 1, 1)以便广播
        onehot_targets = onehot_targets.view(-1, self.num_classes, 1, 1)
        
        # 创建与图像同尺寸的条件mask (batch_size, num_classes, height, width)
        ones = torch.ones(features.size()[0], 
                         self.num_classes,
                         features.size()[2], 
                         features.size()[3], 
                         dtype=features.dtype).to(device)
        # 通过乘法将条件信息广播到图像空间 (batch_size, num_classes, height, width)
        ones = ones * onehot_targets
        # 拼接图像和条件信息 (batch_size, 1+num_classes, height, width)
        x = torch.cat((features, ones), dim=1)
        
        ### 特征提取 ###
        # 三层卷积网络逐步下采样
        x = self.enc_conv_1(x)
        x = F.leaky_relu(x)  # 使用LeakyReLU激活函数
        
        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        
        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        
        ### 潜在空间参数估计 ###
        # 展平特征图 (batch_size, 64*2*2)
        flat_features = x.view(-1, 64*2*2)
        # 计算潜在空间均值和对数方差
        z_mean = self.z_mean(flat_features)
        z_log_var = self.z_log_var(flat_features)
        # 通过重参数化技巧采样潜在变量(batch_size, 64*2*2)
        encoded = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded, targets):
        """解码器前向传播过程，将潜在变量和条件标签解码为重构图像
        
        Args:
            encoded: 潜在变量张量，形状为(batch_size, num_latent)
            targets: 类别标签张量，形状为(batch_size,)
            
        Returns:
            decoded: 重构图像张量，形状为(batch_size, 1, height, width)
        """
        ### 条件信息处理 ###
        # 将标签转为one-hot编码 (batch_size, num_classes)
        onehot_targets = to_onehot(targets, self.num_classes, device)
        # 拼接潜在变量和条件信息 (batch_size, num_latent+num_classes)
        encoded = torch.cat((encoded, onehot_targets), dim=1)
        
        ### 特征重建 ###
        # 全连接层扩展维度 (batch_size, 64*2*2)
        x = self.dec_linear_1(encoded)
        # 调整形状为(batch_size, 64, 2, 2)以匹配转置卷积输入
        x = x.view(-1, 64, 2, 2)
        
        ### 图像重建 ###
        # 三层转置卷积逐步上采样
        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)  # 使用LeakyReLU激活函数
        
        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        
        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)
        
        # 使用sigmoid将输出限制在[0,1]范围，匹配MNIST像素值
        decoded = torch.sigmoid(x)
        
        return decoded

    def forward(self, features, targets):
        
        z_mean, z_log_var, encoded = self.encoder(features, targets)
        decoded = self.decoder(encoded, targets)
        
        return z_mean, z_log_var, encoded, decoded

    
torch.manual_seed(random_seed)
model = ConditionalVariationalAutoencoder(num_features,
                                          num_latent,
                                          num_classes)
model = model.to(device)
    

##########################
### COST AND OPTIMIZER
##########################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

start_time = time.time()

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        z_mean, z_log_var, encoded, decoded = model(features, targets)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + 
                                torch.exp(z_log_var) - z_log_var - 1)).sum()
        
        
        ### Add condition
        onehot_targets = to_onehot(targets, num_classes, device)
        onehot_targets = onehot_targets.view(-1, num_classes, 1, 1)
        
        ones = torch.ones(features.size()[0], 
                          num_classes,
                          features.size()[2], 
                          features.size()[3], 
                          dtype=features.dtype).to(device)
        ones = ones * onehot_targets
        x_con = torch.cat((features, ones), dim=1)
        
        # 解码器的输出为11通道(11=1图像+10条件)，维度匹配
        ### Compute loss
        pixelwise_bce = F.binary_cross_entropy(decoded, x_con, reduction='sum')
        cost = kl_divergence + pixelwise_bce
        
        ### UPDATE MODEL PARAMETERS
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

import matplotlib.pyplot as plt

##########################
### VISUALIZATION
##########################

n_images = 15
image_width = 28

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                         sharex=True, sharey=True, figsize=(20, 2.5))
orig_images = features[:n_images]
decoded_images = decoded[:n_images, 0]

for i in range(n_images):
    for ax, img in zip(axes, [orig_images, decoded_images]):
        ax[i].imshow(img[i].detach().to(torch.device('cpu')).reshape((image_width, image_width)), cmap='binary')

for i in range(10):

    ##########################
    ### RANDOM SAMPLE
    ##########################    
    
    labels = torch.tensor([i]*10).to(device)
    n_images = labels.size()[0]
    rand_features = torch.randn(n_images, num_latent).to(device)
    new_images = model.decoder(rand_features, labels)

    ##########################
    ### VISUALIZATION
    ##########################

    image_width = 28

    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(10, 2.5), sharey=True)
    decoded_images = new_images[:n_images, 0]

    print('Class Label %d' % i)

    for ax, img in zip(axes, decoded_images):
        ax.imshow(img.detach().to(torch.device('cpu')).reshape((image_width, image_width)), cmap='binary')
        
    plt.show()