  # @file    mlp_batchnorm.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/31 21:26:16
  # @version 1.0
  # @brief 

import time
import numpy as np
import torch.backends
import torch.backends.cudnn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.1
num_epochs = 10
batch_size = 64

# Architecture
num_features = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10

# Dataset
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

for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultilayerPerceptron, self).__init__()

        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        # 层归一化是对隐藏层进行归一化
        # BatchNorm1d 会对每个特征维度（num_features）单独计算当前 batch 的均值和方差，然后归一化。
        self.linear_1_bn = torch.nn.BatchNorm1d(num_hidden_1)
        
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        self.linear_2_bn = torch.nn.BatchNorm1d(num_hidden_2)
        
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self,x):
        out = self.linear_1(x)
        out = self.linear_1_bn(out)
        out = F.relu(out)
        out = self.linear_2(out)
        out = self.linear_2_bn(out)
        out = F.relu(out)
        logits = self.linear_out(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

torch.manual_seed(random_seed)
model = MultilayerPerceptron(num_features=num_features, num_classes=num_classes)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

def compute_accuracy(net,data_loader):
    net.eval()
    correct_pred,num_examples =0,0
    with torch.no_grad():
        for features,targets in data_loader:
            features=features.view(-1,28*28).to(device)
            targets = targets.to(device)
            logits,probas = net(features)
            _,predicted_labels = torch.max(probas,1)
            num_examples+=targets.size(0)
            correct_pred+=(predicted_labels==targets).sum()
            return correct_pred.float()/num_examples*100
        
start_time =time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx,(features,targets) in enumerate(train_loader):
        features = features.view(-1,28*28).to(device)
        targets = targets.to(device)
        logits,probas = model(features)
        cost =F.cross_entropy(logits,targets)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch+1, num_epochs, batch_idx, len(train_loader), cost))
            
    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
          epoch+1, num_epochs, 
          compute_accuracy(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))