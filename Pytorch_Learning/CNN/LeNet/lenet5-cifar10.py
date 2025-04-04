  # @file    lenet5-cifar10.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/04 17:44:10
  # @version 1.0
  # @brief 

  # @file    lenet5-mnist.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/04 17:13:40
  # @version 1.0
  # @brief lenet使用tanh作为激活函数

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

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 32*32
NUM_CLASSES = 10

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GRAYSCALE = False

resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])

train_dataset = datasets.CIFAR10(
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
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=8, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         num_workers=8,
                         shuffle=False)
# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

device = torch.device(DEVICE)
# 设置随机数种子（如 torch.manual_seed(0)）的目的是为了 ​​保证实验的可重复性​​。通过固定随机数生成器的初始状态
torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)
        break

class LeNet5(nn.Module):
    def __init__(self,num_classes,grayscale = False):
        super(LeNet5,self).__init__()
        self.grayscale = grayscale
        self.num_classes = num_classes
        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,6*in_channels,kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6*in_channels,16*in_channels,kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5*in_channels,120*in_channels),
            nn.Tanh(),
            nn.Linear(120*in_channels,84*in_channels),
            nn.Tanh(),
            nn.Linear(84*in_channels,self.num_classes) 
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        logits = self.classifier(x)
        probas = F.softmax(logits,dim=1)
        return logits,probas

torch.manual_seed(RANDOM_SEED)
model = LeNet5(NUM_CLASSES,GRAYSCALE)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

def compute_accuracy(model,data_loader,device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model = model.train()
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
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch + 1, NUM_EPOCHS, batch_idx,
                     len(train_loader), cost))
            
    model = model.eval()
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch + 1, NUM_EPOCHS,
              compute_accuracy(model, train_loader, device=DEVICE)))
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

with torch.set_grad_enabled(False):  # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))

for batch_idx, (features, targets) in enumerate(test_loader):
    features = features
    targets = targets
    break

nhwc_img = np.transpose(features[0], axes=(1, 2, 0))
nhw_img = np.squeeze(nhwc_img.numpy(), axis=2)
plt.imshow(nhw_img, cmap='Greys')
model = model.eval()
logits, probas = model(features.to(device)[0, None])
print('Probability %.2f %%' % (probas[0][7] * 100))