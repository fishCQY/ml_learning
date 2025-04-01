# @file    CNN-basic.py
# @author  cqy 3049623863@qq.com
# @date    2025/04/01 20:37:49
# @version 1.0
# @brief Convolutional NeuraL Network (CNN) from scratch using PyTorch.scratch 手动实现
# CNN的原理：通过卷积、激活、池化等操作逐层提取抽象特征，最终通过全连接层完成分类/回归。
# 卷积层：提取局部特征，减少参数数量。
# 激活层：引入非线性变换，增强模型表达能力。
# 池化层：降低特征图尺寸，减少计算量。
# 全连接层：将特征图展平，进行分类/回归。
# 损失函数：衡量模型预测与真实标签的差异。
# 优化器：更新模型参数，减小损失。
# 训练流程：前向传播、计算损失、反向传播、更新参数。
# 应用场景：图像分类、目标检测、自然语言处理等。
# 注意事项：数据预处理、模型复杂度、过拟合等问题。
import time
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Settings and Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.05
num_epochs = 10
batch_size = 128

# Architecture
num_classes = 10

# MNist dataset (images and labels)
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
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


# Model
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # calculate same padding:填充后输出尺寸与输入尺寸相同
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        # w：输入特征图的尺寸（宽度或高度）。
        # k：卷积核（Kernel）的尺寸。
        # p：填充（Padding）量。
        # s：步长（Stride）。
        # o：输出特征图的尺寸。
        # Padding的作用：保持特征图尺寸、保留边缘信息、平衡网络深度与细节。
        # 28x28x1 => 28x28x8

        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(28-1) - 28 + 3) / 2 = 1
        # 28x28x8 => 14x14x8
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)  # (2(14-1) - 14 + 2) / 2 = 0
        # 14x14x8 => 14x14x16
        self.conv_2 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(14-1) - 14 + 3) / 2 = 1
        # 14x14x16 => 7x7x16
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)  # (2(7-1) - 7 + 2) / 2 = 0
        self.linear_1 = torch.nn.Linear(7 * 7 * 16, num_classes)

        # optionally initialize weights from Gaussian;
        # Guassian weight init is not recommended and only for demonstration purposes

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)

        logits = self.linear_1(out.view(-1, 7 * 7 * 16))
        probas = F.softmax(logits, dim=1)
        return logits, probas


torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float() / num_examples * 100


start_time = time.time()
for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)

        # Forward and Backward Passes
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()

        # Update model parameters
        optimizer.step()

        # Logging
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx, len(train_loader), cost))

    model = model.eval()  # eval model after training
    print('Epoch: %03d/%03d training accuracy: %.2f%%' %
          epoch + 1, num_epochs, compute_accuracy(model, train_loader))
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

with torch.set_grad_enabled(False):  # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))
