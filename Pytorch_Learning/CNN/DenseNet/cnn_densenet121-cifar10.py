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
learning_rate = 0.001
num_epochs = 20
batch_size = 128

# Architecture
num_classes = 10
gray_scale = False  # 灰度图像，通道数为1

train_indices = torch.arange(0,48000)
valid_indices = torch.arange(48000,50000)

train_and_valid = datasets.CIFAR10(
    root = '/root/autodl-fs/ml_learning/Pytorch_Learning/datasets',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset = datasets.CIFAR10(
    root = '/root/autodl-fs/ml_learning/Pytorch_Learning/datasets',
    train = False,
    transform = transforms.ToTensor(),
    download = False
)
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

# Check that shuffling works properly
# i.e., label indices should be in random order.
# Also, the label order should be different in the second epoch.
for images,labels in train_loader:
    pass
print(labels[:10])
for images,labels in train_loader:
    pass
print(labels[:10])

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
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
        
    
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet121(nn.Module):
    r"""
    Densenet-BC 模型类是基于论文《"Densely Connected Convolutional Networks" https://arxiv.org/pdf/1608.06993.pdf》实现的密集连接卷积网络（Densenet）。
    该模型通过密集连接的方式重新利用特征图，减少参数数量，同时提高了网络的表达能力。
    参数说明
    growth_rate (int): 每个层中增加的滤器数量，即论文中的 k。
    例如：如果 growth_rate=12，则每个密集层会增加 12 个滤器。
    block_config (list of 4 ints): 每个池化块中包含的层的数量。
    例如：block_config=[2, 2, 2, 2] 表示网络包含 4 个池化块，每个块中有 2 个密集层。
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
    def __init__(self,growth_rate = 32,block_config =(6,12,24,16),
                 num_init_featuremaps = 64,bn_size =4,drop_rate =0,
                 num_classes =1000,memory_efficient = False,gray_scale = False):
        super(DenseNet121,self).__init__()
        if gray_scale:
            in_channels =1
        else:
            in_channels =3
        self.features = nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(in_channels=in_channels,out_channels=num_init_featuremaps,
                               kernel_size=7,stride=2,
                               padding =3,bias=False)),
            ('norm0',nn.BatchNorm2d(num_features=num_init_featuremaps)),
            ('relu0',nn.ReLU(inplace=True)),
            ('pool0',nn.MaxPool2d(kernel_size=3,stride=2,padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_featuremaps
        for i,num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d'%(i+1),block)
            num_features = num_features +num_layers*growth_rate
            if i != len(block_config)-1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features//2)
                self.features.add_module('transition%d'%(i+1),trans)
                num_features = num_features //2
        # Final batch norm
        self.features.add_module('norm5',nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features,num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.constant_(m.bias,0)
    def forward(self,x):
        features = self.features(x)
        out = F.relu(features,inplace=True)
        out = F.adaptive_avg_pool2d(out,(1,1))
        out = torch.flatten(out,1)
        logits = self.classifier(out)
        probas = F.softmax(logits,dim=1)
        return logits,probas
    
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