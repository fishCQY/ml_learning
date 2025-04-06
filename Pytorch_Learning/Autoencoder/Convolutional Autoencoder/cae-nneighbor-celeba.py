  # @file    cae-nneighbor-celeba.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/06 10:13:16
  # @version 1.0
  # @brief 

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

df2 = pd.read_csv('list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
df2.columns = ['Filename', 'Partition']
df2 = df2.set_index('Filename')

df2.head()

df2.loc[df2['Partition'] == 0].to_csv('celeba-train.csv')
df2.loc[df2['Partition'] == 1].to_csv('celeba-valid.csv')
df2.loc[df2['Partition'] == 2].to_csv('celeba-test.csv')

img = Image.open('img_align_celeba/000001.jpg')
print(np.asarray(img, dtype=np.uint8).shape)
plt.imshow(img);

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return self.img_names.shape[0]
    

# Note that transforms.ToTensor()
# already divides pixels by 255. internally

custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])

train_dataset = CelebaDataset(csv_path='celeba-gender-train.csv',
                              img_dir='img_align_celeba/',
                              transform=custom_transform)

BATCH_SIZE=128


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 1e-4
num_epochs = 20

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size):
        super(AutoEncoder, self).__init__()
        
        # 网络参数
        self.in_channels = in_channels      # 输入图像的通道数（如RGB为3）
        self.dec_channels = dec_channels   # 解码器基础通道数（控制网络宽度）
        self.latent_size = latent_size      # 潜在空间维度（压缩后的特征长度）

        ###############
        # ENCODER（编码器）
        ##############
        # 层级结构：5层下采样卷积，通道数逐层倍增
        # 输入尺寸假设：64x64（通过5次stride=2的卷积后得到4x4）
        # 计算：64 -> 32 -> 16 -> 8 -> 4 -> 4（最后一次卷积不改变尺寸）
        
        # Conv1: [B,C,64,64] => [B,dec,32,32]
        self.e_conv_1 = nn.Conv2d(in_channels, dec_channels, 
                                 kernel_size=4, stride=2, padding=1)
        self.e_bn_1 = nn.BatchNorm2d(dec_channels)
        
        # Conv2: [B,dec,32,32] => [B,dec*2,16,16]
        self.e_conv_2 = nn.Conv2d(dec_channels, dec_channels*2, 
                                 kernel_size=4, stride=2, padding=1)
        self.e_bn_2 = nn.BatchNorm2d(dec_channels*2)
        
        # Conv3: [B,dec*2,16,16] => [B,dec*4,8,8]
        self.e_conv_3 = nn.Conv2d(dec_channels*2, dec_channels*4, 
                                 kernel_size=4, stride=2, padding=1)
        self.e_bn_3 = nn.BatchNorm2d(dec_channels*4)
        
        # Conv4: [B,dec*4,8,8] => [B,dec*8,4,4]
        self.e_conv_4 = nn.Conv2d(dec_channels*4, dec_channels*8, 
                                 kernel_size=4, stride=2, padding=1)
        self.e_bn_4 = nn.BatchNorm2d(dec_channels*8)
        
        # Conv5: [B,dec*8,4,4] => [B,dec*16,4,4]（stride=2但padding=1不改变尺寸）
        self.e_conv_5 = nn.Conv2d(dec_channels*8, dec_channels*16, 
                                 kernel_size=4, stride=2, padding=1)
        self.e_bn_5 = nn.BatchNorm2d(dec_channels*16)
        
        # 全连接层：将4x4x(dec*16)展平后映射到潜在空间
        self.e_fc_1 = nn.Linear(dec_channels*16 * 4 * 4, latent_size)

        ###############
        # DECODER（解码器）
        ##############
        # 层级结构：通过插值上采样+卷积重建图像
        
        # 全连接层：将潜在向量恢复为4x4x(dec*16)
        self.d_fc_1 = nn.Linear(latent_size, dec_channels*16 * 4 * 4)
        
        # 解码器卷积层（使用普通卷积而非转置卷积）
        # 每层包含：插值上采样 → 填充 → 卷积 → 激活 → BN
        
        # DeConv1: [B,dec*16,4,4] => [B,dec*8,8,8]
        self.d_conv_1 = nn.Conv2d(dec_channels*16, dec_channels*8, 
                                 kernel_size=4, stride=1, padding=0)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels*8)
        
        # DeConv2: [B,dec*8,8,8] => [B,dec*4,16,16]
        self.d_conv_2 = nn.Conv2d(dec_channels*8, dec_channels*4, 
                                 kernel_size=4, stride=1, padding=0)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels*4)
        
        # DeConv3: [B,dec*4,16,16] => [B,dec*2,32,32]
        self.d_conv_3 = nn.Conv2d(dec_channels*4, dec_channels*2, 
                                 kernel_size=4, stride=1, padding=0)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels*2)
        
        # DeConv4: [B,dec*2,32,32] => [B,dec,64,64]
        self.d_conv_4 = nn.Conv2d(dec_channels*2, dec_channels, 
                                 kernel_size=4, stride=1, padding=0)
        self.d_bn_4 = nn.BatchNorm2d(dec_channels)
        
        # 输出层：生成与输入同尺寸的图像
        self.d_conv_5 = nn.Conv2d(dec_channels, in_channels, 
                                 kernel_size=4, stride=1, padding=0)
        
        # 参数初始化（He初始化适配LeakyReLU）
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        # 编码过程：5层卷积下采样
        x = F.leaky_relu(self.e_bn_1(self.e_conv_1(x)), 0.2, True) # [B,dec,32,32]
        x = F.leaky_relu(self.e_bn_2(self.e_conv_2(x)), 0.2, True) # [B,dec*2,16,16]
        x = F.leaky_relu(self.e_bn_3(self.e_conv_3(x)), 0.2, True) # [B,dec*4,8,8]
        x = F.leaky_relu(self.e_bn_4(self.e_conv_4(x)), 0.2, True) # [B,dec*8,4,4]
        x = F.leaky_relu(self.e_bn_5(self.e_conv_5(x)), 0.2, True) # [B,dec*16,4,4]
        x = x.view(-1, self.dec_channels*16 * 4 * 4)                  # 展平 [B, dec*16 * 4 * 4]
        return self.e_fc_1(x)                                      # 压缩到潜在空间 [B, latent_size]

    def decode(self, z):
        # 解码过程：上采样+卷积重建
        x = F.leaky_relu(self.d_fc_1(z), 0.2, True)                # 全连接扩展 [B, dec*16 * 4 * 4]
        x = x.view(-1, self.dec_channels*16, 4, 4)                # 重塑为4x4特征图 [B, dec*16,4,4]
        
        # 上采样至8x8 → 卷积 → 输出[B,dec*8,8,8]
        x = F.interpolate(x, scale_factor=2, mode='nearest')       # 最近邻插值上采样到8x8
        x = F.pad(x, (2,1,2,1), mode='replicate')                 # 填充（左2右1，上2下1）→ 11x11
        x = F.leaky_relu(self.d_bn_1(self.d_conv_1(x)), 0.2, True) # 卷积后尺寸：11-4+1=8x8
        
        # 重复上采样步骤，逐步恢复分辨率
        x = F.interpolate(x, scale_factor=2)                      # 16x16
        x = F.pad(x, (2,1,2,1), mode='replicate')                 # 19x19
        x = F.leaky_relu(self.d_bn_2(self.d_conv_2(x)), 0.2, True) # 16x16
        
        x = F.interpolate(x, scale_factor=2)                      # 32x32
        x = F.pad(x, (2,1,2,1), mode='replicate')
        x = F.leaky_relu(self.d_bn_3(self.d_conv_3(x)), 0.2, True) # 32x32
        
        x = F.interpolate(x, scale_factor=2)                      # 64x64
        x = F.pad(x, (2,1,2,1), mode='replicate')
        x = F.leaky_relu(self.d_bn_4(self.d_conv_4(x)), 0.2, True) # 64x64
        
        # 最终输出层（无BN）
        x = F.interpolate(x, scale_factor=2)                      # 128x128（假设输入为64x64则需调整）
        x = F.pad(x, (2,1,2,1), mode='replicate')
        return torch.sigmoid(self.d_conv_5(x))                     # 输出像素值在[0,1]之间

    def forward(self, x):
        z = self.encode(x)           # 编码得到潜在向量
        decoded = self.decode(z)    # 解码重建图像
        return z, decoded           # 返回潜在向量和重建结果
##########################
### TRAINING
##########################

epoch_start = 1


torch.manual_seed(random_seed)
model = AutoEncoder(in_channels=3, dec_channels=32, latent_size=1000)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


################## Load previous
# the code saves the autoencoder
# after each epoch so that in case
# the training process gets interrupted,
# we will not have to start training it
# from scratch
files = os.listdir()

for f in files:
    if f.startswith('autoencoder_i_') and f.endswith('.pt'):
        print('Load', f)
        epoch_start = int(f.split('_')[-2]) + 1
        model.load_state_dict(torch.load(f))
        break
##################

start_time = time.time()
for epoch in range(epoch_start, num_epochs+1):
    
    
    for batch_idx, features in enumerate(train_loader):

        # don't need labels, only the images (features)
        features = features.to(device)
        
        ### FORWARD AND BACK PROP
        latent_vector, decoded = model(features)
        cost = F.mse_loss(decoded, features)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 500:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch, num_epochs, batch_idx, 
                     len(train_loader), cost))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    # Save model
    if os.path.isfile('autoencoder_i_%d_%s.pt' % (epoch-1, device)):
        os.remove('autoencoder_i_%d_%s.pt' % (epoch-1, device))
    torch.save(model.state_dict(), 'autoencoder_i_%d_%s.pt' % (epoch, device))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


import matplotlib.pyplot as plt


model = AutoEncoder(in_channels=3, dec_channels=32, latent_size=1000)
model = model.to(device)
model.load_state_dict(torch.load('autoencoder_i_20_%s.pt' % device))
model.eval()
torch.manual_seed(random_seed)

for batch_idx, features in enumerate(train_loader):
    features = features.to(device)
    logits, decoded = model(features)
    break



##########################
### VISUALIZATION
##########################

n_images = 5

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                         sharex=True, sharey=True, figsize=(18, 5))
orig_images = features.detach().cpu().numpy()[:n_images]
orig_images = np.moveaxis(orig_images, 1, -1)

decoded_images = decoded.detach().cpu().numpy()[:n_images]
decoded_images = np.moveaxis(decoded_images, 1, -1)


for i in range(n_images):
    for ax, img in zip(axes, [orig_images, decoded_images]):
        ax[i].axis('off')
        ax[i].imshow(img[i])