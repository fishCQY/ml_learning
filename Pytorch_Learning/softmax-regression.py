  # @file    softmax-regression.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/27 14:59:18
  # @version 1.0
  # @brief Implementation of softmax regression(multinomial logistic regression)

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

# Settings and Datasets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparameters
random_seed =123
learning_rate =0.1
num_epochs =10
batch_size =256

#Architecture
num_features =784
num_classes =10

# Minist dataset
# ToTensor() 将图像从PIL格式转为PyTorch张量（形状 [1, 28, 28]），并自动将像素值从0-255缩放到0-1。
# MNIST数据集默认分为 ​60,000张训练图片 和 ​10,000张测试图片。
# 当 train=False 时，加载的是测试集（10,000张图片）。
train_dataset = datasets.MNIST(root='datasets', # 数据存储目录（若不存在会自动创建）
                               train=True,# 加载训练集（共60,000张图片）
                               transform=transforms.ToTensor(),# 将PIL图像转换为张量，并归一化像素值到[0,1]
                               download=True)# 如果本地无数据，自动从网络下载
test_dataset = datasets.MNIST(root='datasets',
                              train=False,# 加载测试集（共10,000张图片）
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,# 训练数据集对象
                          batch_size=batch_size,# 每个批次的样本数（需提前定义batch_size变量）
                          shuffle=True) # 每个epoch打乱数据顺序，防止模型记忆样本顺序
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)# 测试集无需打乱顺序

#Checking the dataset
# 检查第一个批次的形状
for images, labels in train_loader:
    print('image batch dimensions:', images.shape)  # 输出图像批次形状
    print('image label dimensions:', labels.shape)   # 输出标签批次形状
    break  # 只看第一个批次
#model
class SoftmaxRegression(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        """初始化softmax回归模型参数
        参数:
            num_features (int): 输入特征维度（MNIST为28x28=784）
            num_classes (int): 输出类别数（MNIST为0-9共10类）
        功能:
            1. 继承nn.Module基类
            2. 创建线性变换层 z = Wx + b
            3. 显式零初始化参数（演示用，实际推荐Xavier初始化）
        """
        super(SoftmaxRegression,self).__init__()  # 必须的父类初始化
        self.linear = torch.nn.Linear(num_features,num_classes)  # 创建线性层 [784×10]
        
        # 显式零初始化参数
        self.linear.weight.detach().zero_()  # 权重矩阵 [num_features×num_classes]
        self.linear.bias.detach().zero_()    # 偏置项 [num_classes]
    def forward(self,x):
        logits = self.linear(x) #logits 形状为 [batch_size, num_classes]，则每行（每个样本）的概率和为1。​
        probas = F.softmax(logits,dim=1)# dim=1：指定在类别维度（第1维）上进行归一化
        return logits,probas
    
model = SoftmaxRegression(num_features=num_features,num_classes=num_classes)
model.to(device)

# cost and optimizer
# 传入模型参数和学习率
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# Manual seed for deterministic data loader
torch.manual_seed(random_seed)

def compute_accuracy(model,data_loader):
    correct_pred,num_examples=0,0 # correct_pred：记录正确预测的样本数。num_examples：记录总样本数
    for features,targets in data_loader: # features：一个批次的输入数据（如图像张量）。
        features = features.view(-1,28*28).to(device)# 将输入图像展平为一维向量（假设原始图像形状为 [batch_size, 1, 28, 28]，展平后为 [batch_size, 784]）
        targets = targets.to(device)
        logits,probas = model(features)# logits**：模型原始输出（未归一化的预测值）。probas**：经过Softmax处理后的概率分布（每行和为1）
        _,predicted_labels = torch.max(probas,1)#_：忽略最大值（概率）*predicted_labels**：预测的类别标签（形状为 [batch_size]）
        num_examples+=targets.size(0)
        correct_pred+=(predicted_labels==targets).sum()

    return correct_pred.float()/num_examples*100
for epoch in range(num_epochs):
    for batch_idx,(features,targets) in enumerate(train_loader):
        features = features.view(-1,28*28).to(device)
        targets = targets.to(device)

        # Forward and back prop
        logits,probas = model(features)
        # 这里输入logits是因为交叉熵损失包含了softmax
        #CrossEntropy(y,p)=− CrossEntropy(y,p)=−∑ylog(p)
        # 由于真实标签 y 是one-hot编码（仅有一个 yk=1，其余为0）上式简化为：Loss =−log(p k)
        cost = F.cross_entropy(logits,targets)
        optimizer.zero_grad()
        cost.backward()
        # Update model parameters
        optimizer.step()

        # Logging 每50个批次输出一次数据
        # len(train_dataset)//batch_size获取总的批次数
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                     %(epoch+1,num_epochs,batch_idx,
                     len(train_dataset)//batch_size,cost))
            
        with torch.set_grad_enabled(False): # 在不需要梯度的代码块中禁用梯度计算，提升性能和节省内存。
            print('Epoch: %03d/%03d training accuaracy: %.2f%%' % 
                  (epoch+1,num_epochs,
                   compute_accuracy(model,train_loader)))
            
print('test accuracy: %.2f%%' % (compute_accuracy(model,test_loader)) )

            
            
        