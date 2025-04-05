  # @file    autoencoder-randomforest.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/05 21:50:23
  # @version 1.0
  # @brief 

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 5
batch_size = 256

# Architecture
num_features = 784
num_hidden_1 = 32


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

class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        
        ### ENCODER
        
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # The following to lones are not necessary, 
        # but used here to demonstrate how to access the weights
        # and use a different weight initialization.
        # By default, PyTorch uses Xavier/Glorot initialization, which
        # should usually be preferred.
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        
        ### DECODER
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        
    def encoder(self, x):
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        return encoded
    
    def decoder(self, encoded_x):
        logits = self.linear_2(encoded_x)
        decoded = torch.sigmoid(logits)
        return decoded
        

    def forward(self, x):
        
        ### ENCODER
        encoded = self.encoder(x)
        
        ### DECODER
        decoded = self.decoder(encoded)
        
        return decoded

    
torch.manual_seed(random_seed)
model = Autoencoder(num_features=num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        # don't need labels, only the images (features)
        features = features.view(-1, 28*28).to(device)
            
        ### FORWARD AND BACK PROP
        decoded = model(features)
        cost = F.binary_cross_entropy(decoded, features)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
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


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=15, 
                          shuffle=True)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
    
# =============================================================

n_images = 15
image_width = 28

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                         sharex=True, sharey=True, figsize=(20, 2.5))
orig_images = features[:n_images]
decoded_images = decoded[:n_images]

for i in range(n_images):
    for ax, img in zip(axes, [orig_images, decoded_images]):
        curr_img = img[i].detach().to(torch.device('cpu'))
        ax[i].imshow(curr_img.view((image_width, image_width)), cmap='binary')

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=15, 
                         shuffle=True)

# Checking the dataset
for images, labels in test_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

import numpy as np
from sklearn.ensemble import RandomForestClassifier


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=60000, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=10000, 
                          shuffle=False)
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

X_train = np.array(images.reshape(60000, 28*28))
y_train = np.array(labels)


for images, labels in test_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

X_test = np.array(images.reshape(10000, 28*28))
y_test = np.array(labels)

# 创建并训练随机森林分类器
# n_estimators=500 表示使用500棵决策树
# n_jobs=-1 表示使用所有CPU核心并行计算
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(X_train, y_train)

# 在训练集上评估模型准确率
# score方法计算模型在给定数据上的分类准确率
# f-string格式化输出百分比结果
print(f'Train Accuracy: {rf.score(X_train, y_train)*100}%')

# 在测试集上评估模型准确率
print(f'Test Accuracy: {rf.score(X_test, y_test)*100}%')

from sklearn.decomposition import PCA

# 创建PCA降维器，将784维特征压缩到32维（与自编码器潜在空间维度(隐藏层)相同）
pca = PCA(n_components=32)

# 对训练集进行PCA降维（同时学习降维参数）
X_train_pca = pca.fit_transform(X_train)  # X_train形状(60000,784) -> (60000,32)

# 对测试集应用相同的降维变换（不重新学习参数）
X_test_pca = pca.transform(X_test)  # X_test形状(10000,784) -> (10000,32)

# 创建并训练随机森林分类器
# n_estimators=500 表示使用500棵决策树
# n_jobs=-1 表示使用所有CPU核心并行计算
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(X_train_pca, y_train)

# 评估训练集准确率（降维后数据）
print(f'Train Accuracy: {rf.score(X_train_pca, y_train)*100}%')

# 评估测试集准确率（降维后数据）
print(f'Test Accuracy: {rf.score(X_test_pca, y_test)*100}%')

# compressed mnist
# 创建训练数据加载器，每批1000个样本
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=1000, 
                         shuffle=True)  # 打乱数据顺序

# 创建测试数据加载器，每批1000个样本
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=1000,
                        shuffle=False)  # 测试集不需要打乱

# 初始化训练集特征矩阵(60000个样本，每个样本num_hidden_1维)
X_train_compr = np.ones((60000, num_hidden_1))  # num_hidden_1=32（编码器输出维度）
# 初始化训练集标签向量
y_train = np.ones(60000)

start_idx = 0  # 记录当前填充位置

# 分批处理训练数据
for idx, (images, labels) in enumerate(train_loader):
    # 将图像展平为784(28x28)维向量并送入GPU
    features = images.view(-1, 28*28).to(device)
    
    # 使用编码器压缩特征(784D -> 32D)
    decoded = model.encoder(features)
    
    # 将压缩后的特征转回CPU并转为numpy数组，存入对应位置
    X_train_compr[start_idx:start_idx+1000] = decoded.to('cpu').detach().numpy()
    # 存储对应的标签
    y_train[start_idx:start_idx+1000] = labels
    
    # 更新填充位置
    start_idx += 1000

# 初始化测试集特征矩阵(10000个样本，每个样本32维)
X_test_compr = np.ones((10000, num_hidden_1))
# 初始化测试集标签向量
y_test = np.ones(10000)

start_idx = 0

for idx, (images, labels) in enumerate(test_loader): 
    features = images.view(-1, 28*28).to(device)
    decoded = model.encoder(features)
    X_test_compr[start_idx:start_idx+1000] = decoded.to(torch.device('cpu')).detach().numpy()
    y_test[start_idx:start_idx+1000] = labels
    start_idx += 1000

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(X_train_compr, y_train)
print(f'Train Accuracy: {rf.score(X_train_compr, y_train)*100}%')
print(f'Test Accuracy: {rf.score(X_test_compr, y_test)*100}%')