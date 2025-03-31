  # @file    mlp_dropout.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/31 22:00:33
  # @version 1.0
  # @brief 

import time 
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True #设置cuDNN后端使用确定性算法，每次运行都会得到相同的计算结果


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.1
num_epochs = 10
batch_size = 64
dropout_prob =0.5 #dropout概率

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
                          shuffle=True)  # 只在训练的时候打乱
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

        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2) 
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()

        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self,x):
        out = self.linear_1(x)
        out = F.relu(out)
        out = F.dropout(out,p=dropout_prob,training=self.training) #training=self.training 表示在训练时使用dropout，在测试时不使用dropout。
        out = self.linear_2(out)
        out = F.relu(out)
        out = F.dropout(out,p=dropout_prob,training=self.training)
        logits = self.linear_out(out)
        probas = F.softmax(logits,dim=1)
        return logits,probas
    
torch.manual_seed(random_seed) #设置随机种子，保证每次运行的结果都是一样的
model = MultilayerPerceptron(num_features=num_features,num_classes=num_classes)

model = model.to(device) #将模型移动到GPU上

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) #定义优化器，使用随机梯度下降算法，学习率为0.1

def compute_accuracy(net,data_loader): #计算准确率
    net.eval() #将模型设置为评估模式，不使用dropout
    correct_pred,num_examples =0,0 #初始化正确预测的数量和样本数量
    with torch.no_grad(): #不计算梯度，节省内存
        for features,targets in data_loader: #遍历测试集
            features=features.view(-1,28*28).to(device) #将输入数据展平为一维向量，并移动到GPU上
            targets = targets.to(device) #将标签移动到GPU上
            logits,probas = net(features) #前向传播，得到输出
            _,predicted_labels = torch.max(probas,1) #得到预测的标签
            num_examples+=targets.size(0) #更新样本数量
            correct_pred+=(predicted_labels==targets).sum() #更新正确预测的数量
            return correct_pred.float()/num_examples*100 #返回准确率
        
start_time =time.time() #记录开始时间
for epoch in range(num_epochs): #训练num_epochs轮
    model.train() #将模型设置为训练模式，使用dropout
    for batch_idx,(features,targets) in enumerate(train_loader): #遍历训练集
        features = features.view(-1,28*28).to(device) #将输入数据展平为一维向量，并移动到GPU上
        targets = targets.to(device) #将标签移动到GPU上
        logits,probas = model(features) #前向传播，得到输出
        cost = F.cross_entropy(logits,targets) #计算损失
        optimizer.zero_grad() #清零梯度
        cost.backward() #反向传播
        optimizer.step() #更新参数
        if not batch_idx % 50: #每50个batch输出一次信息
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch+1, num_epochs, batch_idx, len(train_loader), cost))
    
    print('Epoch: %03d/%03d training accuracy: %.2f%%' %
          (epoch+1,num_epochs,compute_accuracy(model,train_loader)))
    
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60)) #输出训练时间
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60)) #输出总训练时间

print('Test accuracy: %.2f%%' % (compute_accuracy(model,test_loader))) #输出测试集上的准确率