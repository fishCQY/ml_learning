  # @file    cnn_densenet121-mnist.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/04 09:50:07
  # @version 1.0
  # @brief 

import os
import time
import numpy as np
import pandas as pd
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True  # 设置cuDNN后端使用确定性算法，每次运行都会得到相同的计算结果

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.0001
num_epochs = 10
batch_size = 128

# Architecture
num_classes = 10
gray_scale = True  # 灰度图像，通道数为1

train_indices = torch.arange(0,59000)
valid_indices = torch.arange(59000,60000)
resize_transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

train_and_valid = datasets.MNIST(
    root = '/root/autodl-fs/ml_learning/Pytorch_Learning/datasets',
    train = True,
    transform = resize_transform,
    download = False
)
test_dataset = datasets.MNIST(
    root = '/root/autodl-fs/ml_learning/Pytorch_Learning/datasets',
    train = False,
    transform = resize_transform,
    download = False
)
# Subset是PyTorch中的一个类，用于从数据集中选取指定的子集
train_dataset = Subset(train_and_valid, train_indices)
valid_dataset = Subset(train_and_valid, valid_indices)

train_loader = DataLoader(dataset = train_dataset, 
                          batch_size = batch_size,
                          num_workers=4,
                          shuffle = True)
valid_loader = DataLoader(dataset = valid_dataset,
                          batch_size = batch_size,
                          num_workers=4,
                          shuffle = False)
test_loader = DataLoader(dataset = test_dataset,
                         batch_size = batch_size,
                         num_workers=4,
                         shuffle = False)

device = torch.device(device)
torch.manual_seed(0)

for epoch in range(2):
    for batch_idx,(x,y) in enumerate(train_loader):
        print('Epoch:',epoch+1,end='')# end = '' 表示不换行
        print('| Batch index:',batch_idx,end='')
        print('| Batch size:',y.size()[0])
        x = x.to(device)
        y = y.to(device)
        break
# ​​如果shuffle=True​​：两次打印的标签顺序​​应该不同​​,验证数据集是否被shuffle
# ​​如果shuffle=False​​：两次打印的标签顺序​​完全相同​​
# 第一次遍历所有数据（触发DataLoader的shuffle机制）
for images, labels in train_loader:
    pass  # 什么也不做，但会触发DataLoader的数据加载流程
print(labels[:10])  # 打印第一个batch的前10个标签

# 第二次遍历所有数据（触发新的shuffle）
for images, labels in train_loader:
    pass
print(labels[:10])  # 再次打印第一个batch的前10个标签

# The following code cell that implements the DenseNet-121 architecture 
# is a derivative of the code provided at 
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as cp

def _bn_function_factory(norm, relu, conv):
    """创建用于特征拼接+BN+激活+卷积的闭包函数
    
    Args:
        norm: 归一化层 (如BatchNorm2d)
        relu: 激活层 (如ReLU) 
        conv: 卷积层 (如Conv2d)
        
    Returns:
        bn_function: 处理输入特征的函数
    """
    def bn_function(*inputs):
        """特征拼接与处理函数
        
        输入: 多个特征图张量
        输出: 处理后的特征张量
        """
        # 沿通道维度拼接所有输入特征
        concated_features = torch.cat(inputs, 1)  # shape: [B, sum(C_i), H, W]
        
        # 标准化 -> 激活 -> 卷积压缩
        bottleneck_output = conv(relu(norm(concated_features)))  
        return bottleneck_output
    
    return bn_function  # 返回配置好的处理函数

class _DenseLayer(nn.Sequential):
    """DenseNet 基础层模块，包含Bottleneck结构
    
    Args:
        num_input_features: 输入通道数
        growth_rate: 每个层输出的新特征图数 (k)
        bn_size: 瓶颈层通道放大因子 (默认4)
        drop_rate: Dropout概率
        memory_efficient: 是否启用内存优化模式
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        
        # 构建Bottleneck结构 (BN -> ReLU -> 1x1Conv -> BN -> ReLU -> 3x3Conv)
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))  # 输入归一化
        self.add_module('relu1', nn.ReLU(inplace=True))               # 激活
        self.add_module('conv1', nn.Conv2d(                           # 1x1卷积压缩通道
            in_channels=num_input_features,
            out_channels=bn_size * growth_rate,  # 压缩到 bn_size*k 通道
            kernel_size=1,
            stride=1,
            bias=False))
        
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))  # 瓶颈层归一化
        self.add_module('relu2', nn.ReLU(inplace=True))                   # 激活
        self.add_module('conv2', nn.Conv2d(                               # 3x3卷积生成新特征
            in_channels=bn_size * growth_rate,
            out_channels=growth_rate,  # 最终输出k个特征图
            kernel_size=3,
            stride=1,
            padding=1,  # 保持空间尺寸不变
            bias=False))
        
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient  # 是否启用梯度检查点节省内存

    def forward(self, *prev_features):
        """前向传播，处理所有先前层的特征
        
        输入: 来自前面所有层的特征图列表 
        输出: 新生成的特征图
        """
        # 使用工厂函数生成特征处理函数
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        
        # 内存高效模式：使用梯度检查点减少内存占用
        # self.memory_efficient​​：布尔值，控制是否启用内存优化模式
        # ​​any(prev_feature.requires_grad ...)​​：检查输入特征中是否有需要梯度计算的张量
        # cp.checkpoint(bn_function, *prev_features)​​
        # ​​cp.checkpoint​​：PyTorch 的梯度检查点功能
        # ​​作用​​：在前向传播时不保存中间激活值，而是在反向传播时重新计算
        # ​​节省显存​​：减少约 30%-50% 的显存占用
        # ​​代价​​：增加约 20% 的计算时间
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)  # 分段计算，节省内存
        else:
            bottleneck_output = bn_function(*prev_features)  # 常规前向计算
            
        # 通过第二组BN+ReLU+Conv
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))# 所谓的H函数
        
        # 应用Dropout（如果启用）
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            
        return new_features  # 返回当前层生成的新特征
        
    
class _DenseBlock(nn.Module):
    """DenseNet 密集块，包含多个密集层
    
    Args:
        num_layers: 当前块中的层数
        num_input_features: 初始输入通道数
        bn_size: 瓶颈层通道放大因子
        growth_rate: 每层生成的新特征数 (k)
        drop_rate: Dropout概率
        memory_efficient: 是否启用内存优化
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        
        # 逐层构建密集层
        for i in range(num_layers):
            # 计算当前层的输入通道数 = 初始输入 + 前面所有层输出的总和
            layer_input_channels = num_input_features + i * growth_rate
            
            # 创建密集层并添加到模块
            layer = _DenseLayer(
                num_input_features=layer_input_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.add_module('denselayer%d' % (i + 1), layer)  # 命名如 denselayer1, denselayer2...

    def forward(self, init_features):
        """前向传播，处理初始特征并逐层生成新特征
        
        输入: 
            init_features: 初始输入特征 (来自前一过渡层或输入)
        输出:
            所有层特征拼接后的结果
        """
        features = [init_features]  # 初始化特征列表
        
        # 逐层处理：每个层接收前面所有层的输出
        for name, layer in self.named_children():
            new_features = layer(*features)  # 将当前所有特征传递给下一层
            features.append(new_features)    # 将新特征加入列表
            
        # 沿通道维度拼接所有层的输出
        return torch.cat(features, dim=1)  # shape: [B, C_init + num_layers*k, H, W]

class _Transition(nn.Sequential):
    """DenseNet的过渡层模块，用于压缩特征图尺寸和通道数
    包含: BN -> ReLU -> 1x1卷积 -> 2x2平均池化
    
    Args:
        num_input_features: 输入特征通道数
        num_output_features: 输出特征通道数（通常为输入的一半）
    """
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # 批量归一化层（处理输入特征）
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        # ReLU激活函数（inplace操作节省内存）
        self.add_module('relu', nn.ReLU(inplace=True))
        # 1x1卷积压缩通道数（降维作用）
        self.add_module('conv', nn.Conv2d(
            num_input_features, 
            num_output_features,
            kernel_size=1, 
            stride=1, 
            bias=False))  # 无偏置项
        # 2x2平均池化下采样（空间维度减半）
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet121(nn.Module):
    r"""
    Densenet-BC 模型类是基于论文《"Densely Connected Convolutional Networks" https://arxiv.org/pdf/1608.06993.pdf》实现的密集连接卷积网络（Densenet）。
    该模型通过密集连接的方式重新利用特征图，减少参数数量，同时提高了网络的表达能力。
    参数说明
    growth_rate (int): 每个层中增加的滤器数量，即论文中的 k。
    例如：如果 growth_rate=12，则每个密集层会增加 12 个滤器。
    block_config (list of 4 ints): 每个密集块中包含的层的数量。
    例如：block_config=[2, 2, 2, 2] 表示网络包含 4 个密集块，每个块中有 2 个密集层。
    num_init_featuremaps (int): 第一层卷积层中学习的滤器数量。
    例如：num_init_featuremaps=64 表示第一层卷积层会输出 64 个特征图。
    bn_size (int): 瓶颈层中滤器数量的乘数因子。
    瓶颈层的滤器数量为 bn_size * growth_rate。
    例如：如果 bn_size=4 和 growth_rate=12，则瓶颈层会有 4 * 12 = 48 个滤器。
    drop_rate (float): 每个密集层后应用的 dropout 率。
    用于随机丢弃部分神经元，防止过拟合。
    例如：drop_rate=0.2 表示每个密集层后随机丢弃 20% 的神经元。
    num_classes (int): 分类任务的类别数量。
    用于定义最后的全连接层的输出大小。
    memory_efficient (bool): 是否使用内存高效模式（检查点）。
    如果 True，会使用检查点技术以节省内存，但会增加计算时间。
    默认值为 False。
    更多细节请参考论文《"Densely Connected Convolutional Networks" https://arxiv.org/pdf/1707.06990.pdf》。
    """
    def __init__(self, growth_rate=32, block_config=(6,12,24,16),
                 num_init_featuremaps=64, bn_size=4, drop_rate=0,
                 num_classes=1000, memory_efficient=False, gray_scale=False):
        """DenseNet121网络结构初始化
        参数:
            growth_rate: 每层新增的特征图数量 (k)
            block_config: 各密集块的层数配置 [4个元素的元组]
            num_init_featuremaps: 初始卷积层输出的特征图数
            bn_size: 瓶颈层通道放大因子
            drop_rate: dropout概率
            gray_scale: 是否为灰度输入 (MNIST需设为True)
        """
        super(DenseNet121, self).__init__()
        # 确定输入通道数（MNIST灰度图为1通道）
        in_channels = 1 if gray_scale else 3
        
        # 初始特征提取层 (conv7x7 + BN + ReLU + maxpool3x3)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_featuremaps,
                               kernel_size=7, stride=2, padding=3, bias=False)),  # 空间维度减半
            ('norm0', nn.BatchNorm2d(num_init_featuremaps)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),  # 再次空间维度减半
        ]))
        
        # 构建密集块和过渡层
        num_features = num_init_featuremaps
        for i, num_layers in enumerate(block_config):
            # 添加密集块
            self.features.add_module('denseblock%d'%(i+1), _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            ))
            num_features += num_layers * growth_rate  # 更新特征通道数
            
            # 非最后一个块后添加过渡层（压缩通道和空间维度）
            if i != len(block_config)-1:
                self.features.add_module('transition%d'%(i+1),
                    _Transition(num_features, num_features//2))  # 通道数减半
                num_features = num_features // 2
                
        # 最终处理层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # 分类器 (全局平均池化后接全连接层)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)  # 卷积层使用He初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)     # BN层gamma初始化为1
                nn.init.constant_(m.bias, 0)        # BN层beta初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)        # 全连接层偏置初始化为0
    def forward(self, x):
        # 前向传播流程
        features = self.features(x)          # 通过特征提取主干网络
        out = F.relu(features, inplace=True) # 应用ReLU激活（原位操作节省内存）
        out = F.adaptive_avg_pool2d(out, (1, 1))  # 全局平均池化到1x1空间维度
        out = torch.flatten(out, 1)          # 展平为[batch_size, features]形状
        logits = self.classifier(out)        # 通过全连接层得到原始分数输出
        probas = F.softmax(logits, dim=1)    # 计算类别概率分布
        return logits, probas                 # 同时返回原始分数和概率（便于训练和推理）
    
torch.manual_seed(random_seed)
model = DenseNet121(num_classes=num_classes,gray_scale=gray_scale)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

def compute_acc(model,data_loader,device):
    model.eval()
    correct_pred,num_examples = 0,0
    for i,(features,targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        logits,probas = model(features)
        _,predicted_labels = torch.max(probas,1)
        num_examples +=targets.size(0)
        assert predicted_labels.size() ==  targets.size()
        correct_pred +=(predicted_labels == targets).sum()
    return correct_pred.float()/num_examples*100
start_time = time.time()
cost_list = []
train_acc_list,valid_acc_list =[],[]

for epoch in range(num_epochs):
    model.train()
    for batch_idx,(features,targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)
        logits,probas = model(features)
        cost = F.cross_entropy(logits,targets)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_list.append(cost.item())
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  %(epoch+1,num_epochs,batch_idx,len(train_loader),cost))
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc = compute_acc(model,train_loader,device=device)
        valid_acc = compute_acc(model,valid_loader,device=device)
        print('Epoch: %03d/%03d | Train ACC: %.3f%% | Valid ACC: %.3f%%'
              %(epoch+1,num_epochs,train_acc,valid_acc))
        
        train_acc_list.append(train_acc.item())
        valid_acc_list.append(valid_acc.item())
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')


plt.plot(cost_list, label='Minibatch cost')
plt.plot(np.convolve(cost_list, 
                     np.ones(200,)/200, mode='valid'), 
         label='Running average')

plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.legend()
plt.show()

plt.plot(np.arange(1, num_epochs+1), train_acc_list, label='Training')
plt.plot(np.arange(1, num_epochs+1), valid_acc_list, label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

with torch.set_grad_enabled(False):
    test_acc = compute_acc(model=model,
                           data_loader=test_loader,
                           device=device)
    valid_acc = compute_acc(model=model,
                            data_loader=valid_loader,
                            device=device)
    print(f'Valid ACC: {valid_acc:.2f}%')
    print(f'Test ACC: {test_acc:.2f}%')