  # @file    resnet18-mnist.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/05 14:48:26
  # @version 1.0
  # @brief 

import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 128*128
NUM_CLASSES = 2
BATCH_SIZE = 256*torch.cuda.device_count()
DEVICE = 'cuda:0' # default GPU device
GRAYSCALE = False


df1 = pd.read_csv('list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male'])

# Make 0 (female) & 1 (male) labels instead of -1 & 1
df1.loc[df1['Male'] == -1, 'Male'] = 0

df1.head()

df2 = pd.read_csv('list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
df2.columns = ['Filename', 'Partition']
df2 = df2.set_index('Filename')

df2.head()

df3 = df1.merge(df2, left_index=True, right_index=True)
df3.head()

df3.to_csv('celeba-gender-partitions.csv')
df4 = pd.read_csv('celeba-gender-partitions.csv', index_col=0)
df4.head()

df4.loc[df4['Partition'] == 0].to_csv('celeba-gender-train.csv')
df4.loc[df4['Partition'] == 1].to_csv('celeba-gender-valid.csv')
df4.loc[df4['Partition'] == 2].to_csv('celeba-gender-test.csv')
img = Image.open('img_align_celeba/000001.jpg')
print(np.asarray(img, dtype=np.uint8).shape)
plt.imshow(img)

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
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

valid_dataset = CelebaDataset(csv_path='celeba-gender-valid.csv',
                              img_dir='img_align_celeba/',
                              transform=custom_transform)

test_dataset = CelebaDataset(csv_path='celeba-gender-test.csv',
                             img_dir='img_align_celeba/',
                             transform=custom_transform)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4)

torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        time.sleep(1)
        break

##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 基础残差块（ResNet-18/34使用）
class BasicBlock(nn.Module):
    expansion = 1  # 通道扩展系数（Bottleneck块中为4）

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """初始化基础残差块
        Args:
            inplanes: 输入通道数
            planes: 基准通道数（实际输出通道为planes * expansion）
            stride: 第一个卷积的步长（控制下采样）
            downsample: 捷径连接的维度匹配模块
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层（可能包含下采样）
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3x3卷积定义见下文
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)  # 原地激活节省内存
        # 第二个卷积层（固定尺寸）
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # 捷径连接的投影模块
        self.stride = stride

    def forward(self, x):
        residual = x  # 原始输入作为残差项

        # 主路径处理流程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 捷径连接处理（维度匹配）
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差相加与激活
        out += residual  # 核心的跳跃连接
        out = self.relu(out)

        return out

# ResNet主体结构
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        """初始化ResNet
        Args:
            block: 残差块类型（BasicBlock/Bottleneck）
            layers: 各阶段残差块数量列表（如[2,2,2,2]）
            num_classes: 分类类别数
            grayscale: 是否灰度输入（控制输入通道）
        """
        self.inplanes = 64  # 初始通道数
        in_dim = 1 if grayscale else 3  # 输入通道自适应
        super(ResNet, self).__init__()
        
        # 初始特征提取层
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差阶段（特征图尺寸逐步下采样）
        self.layer1 = self._make_layer(block, 64, layers[0])      # 输出尺寸56x56
        self.layer2 = self._make_layer(block, 128, layers[1], 2)  # 28x28
        self.layer3 = self._make_layer(block, 256, layers[2], 2)  # 14x14 
        self.layer4 = self._make_layer(block, 512, layers[3], 2)   # 7x7
        
        # 分类头（MNIST输入尺寸小，注释了原设计的平均池化）
        self.avgpool = nn.AvgPool2d(7, stride=1)  # 原设计用于224x224输入
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接分类层

        # 参数初始化策略
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He初始化（适配ReLU）
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层初始化为单位变换，权重初始化为1，偏置初始化为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建残差阶段
        Args:
            block: 残差块类型
            planes: 该阶段基准通道数
            blocks: 包含的残差块数量
            stride: 第一个残差块的步长
        """
        downsample = None
        # 当需要下采样或通道数变化时，构建捷径连接
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个块处理下采样和通道变化
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # 更新输入通道数
        # 添加后续残差块（保持尺寸）
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始特征提取
        x = self.conv1(x)     # 224x224 -> 112x112（原设计）
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # 112x112 -> 56x56
        
        # 通过四个残差阶段
        x = self.layer1(x)    # 56x56
        x = self.layer2(x)    # 28x28
        x = self.layer3(x)    # 14x14
        x = self.layer4(x)    # 7x7
        
        # 对于MNIST（输入28x28），此时特征图已为1x1
        # x = self.avgpool(x)  # 原设计用于获取1x1特征图
        
        x = x.view(x.size(0), -1)  # 展平
        logits = self.fc(x)        # 分类输出
        probas = F.softmax(logits, dim=1)  # 概率分布
        return logits, probas

def resnet18(num_classes):
    """构建ResNet-18实例
    Args:
        num_classes: 输出类别数
    """
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],  # 四个阶段的块数量
                   num_classes=num_classes,
                   grayscale=False)      # 默认RGB输入
    return model


torch.manual_seed(RANDOM_SEED)

##########################
### COST AND OPTIMIZER
##########################



model = resnet18(NUM_CLASSES)


#### DATA PARALLEL START ####
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
#### DATA PARALLEL END ####


model.to(DEVICE)



cost_fn = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=DEVICE)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))

for batch_idx, (features, targets) in enumerate(test_loader):

    features = features
    targets = targets
    break
    
plt.imshow(np.transpose(features[0], (1, 2, 0)))
    
model.eval()
logits, probas = model(features.to(DEVICE)[0, None])
print('Probability Female %.2f%%' % (probas[0][0]*100))