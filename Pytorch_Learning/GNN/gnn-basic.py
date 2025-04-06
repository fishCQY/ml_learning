  # @file    gnn-basic.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/06 21:30:57
  # @version 1.0
  # @brief 

import time
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt

##########################
### SETTINGS
##########################

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.05
NUM_EPOCHS = 20
BATCH_SIZE = 128
IMG_SIZE = 28

# Architecture
NUM_CLASSES = 10

train_indices = torch.arange(0, 59000)
valid_indices = torch.arange(59000, 60000)

custom_transform = transforms.Compose([transforms.ToTensor()])


train_and_valid = datasets.MNIST(root='/root/autodl-fs/ml_learning/Pytorch_Learning/datasets', 
                                 train=True, 
                                 transform=custom_transform,
                                 download=True)

test_dataset = datasets.MNIST(root='/root/autodl-fs/ml_learning/Pytorch_Learning/datasets', 
                              train=False, 
                              transform=custom_transform,
                              download=True)

train_dataset = Subset(train_and_valid, train_indices)
valid_dataset = Subset(train_and_valid, valid_indices)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE,
                         num_workers=4,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
#为图像数据构建图结构的邻接矩阵
# 将图像像素视为图中的节点，基于像素坐标计算节点间连接强度
def precompute_adjacency_matrix(img_size):
    # 生成网格坐标 (img_size x img_size)列坐标矩阵，形状为(img_size, img_size)
    col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
    
    # N = img_size^2
    # 构造2D坐标数组(形状N x 2)并归一化到[0,1]范围
    # stack将行列坐标合并为(x,y)点，reshape展平为N个点
    coord = np.stack((col, row), axis=2).reshape(-1, 2) / img_size

    # 计算所有点对之间的欧式距离矩阵(N x N)
    dist = cdist(coord, coord, metric='euclidean')
    
    # 应用高斯滤波器构建邻接矩阵
    sigma = 0.05 * np.pi  # 高斯核宽度参数，sigma 控制衰减速度，值越小，相似性随距离增加下降越快
    A = np.exp(- dist / sigma ** 2)  # 高斯核函数，将距离转换为相似性，距离越近相似性越高
    A[A < 0.01] = 0  # 阈值处理，移除弱连接
    A = torch.from_numpy(A).float()  # 转换为PyTorch张量

    # 按照Kipf & Welling (ICLR 2017)的方法进行归一化
    D = A.sum(1)  # 计算节点度(N,)
    D_hat = (D + 1e-5) ** (-0.5)  # 度矩阵的-1/2次方(加小常数防止除零)
    # 对称归一化: D^(-1/2) * A * D^(-1/2)
    A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # 结果形状(N,N)
    # A_hat 形状为 784x784，表示784个像素点之间的归一化连接强度。
    
    return A_hat

plt.imshow(precompute_adjacency_matrix(28));

##########################
### MODEL
##########################

        

class GraphNet(nn.Module):
    def __init__(self, img_size=28, num_classes=10):
        super(GraphNet, self).__init__()  # 调用父类nn.Module的初始化
        
        n_rows = img_size**2  # 计算图像展平后的节点数(28x28=784)
        self.fc = nn.Linear(n_rows, num_classes, bias=False)  # 定义全连接层，无偏置项

        # 预计算邻接矩阵(描述图中节点间连接关系)
        A = precompute_adjacency_matrix(img_size)  
        # 将邻接矩阵注册为buffer(不参与训练但会保存到模型状态中)
        self.register_buffer('A', A)

        
    def forward(self, x):
        """图神经网络的前向传播过程
        
        Args:
            x: 输入张量，形状为[B, C, H, W]
                B - batch大小
                C - 通道数(对于MNIST为1)
                H - 图像高度
                W - 图像宽度
                
        Returns:
            tuple: (logits, probas)
                logits - 全连接层输出，形状[B, num_classes]
                probas - softmax后的概率分布，形状[B, num_classes]
        """
        
        B = x.size(0) # 获取batch大小

        ### 邻接矩阵处理
        # 将[N,N]的邻接矩阵扩展为[1,N,N]的张量
        A_tensor = self.A.unsqueeze(0)
        # 将邻接矩阵复制B次，得到[B,N,N]的张量
        A_tensor = self.A.expand(B, -1, -1)
        
        ### 输入数据重塑
        # 将输入从[B,C,H,W]重塑为[B, H*W, 1]B：保持batch维度不变
        # -1：自动计算展平后的维度（这里等于C×H×W=1×28×28=784）
        # 1：新增一个维度，使每个像素值成为单独的节点特征
        x_reshape = x.view(B, -1, 1)
        
        ### 图卷积操作
        # 使用批量矩阵乘法(bmm)聚合邻居特征
        # [B,N,N] x [B,N,1] -> [B,N,1] -> 展平为[B,N]
        avg_neighbor_features = (torch.bmm(A_tensor, x_reshape).view(B, -1))
        
        ### 分类输出，相当于将邻接特征矩阵输入全连接层
        logits = self.fc(avg_neighbor_features)  # 全连接层计算logits
        probas = F.softmax(logits, dim=1)       # softmax得到概率分布
        return logits, probas
    
torch.manual_seed(RANDOM_SEED)
model = GraphNet(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

model = model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  

def compute_acc(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

start_time = time.time()

cost_list = []
train_acc_list, valid_acc_list = [], []


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
        
        #################################################
        ### CODE ONLY FOR LOGGING BEYOND THIS POINT
        ################################################
        cost_list.append(cost.item())
        if not batch_idx % 150:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                   f' Cost: {cost:.4f}')

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        
        train_acc = compute_acc(model, train_loader, device=DEVICE)
        valid_acc = compute_acc(model, valid_loader, device=DEVICE)
        
        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d}\n'
              f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f}')
        
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
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

plt.plot(np.arange(1, NUM_EPOCHS+1), train_acc_list, label='Training')
plt.plot(np.arange(1, NUM_EPOCHS+1), valid_acc_list, label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

with torch.set_grad_enabled(False):
    test_acc = compute_acc(model=model,
                           data_loader=test_loader,
                           device=DEVICE)
    
    valid_acc = compute_acc(model=model,
                            data_loader=valid_loader,
                            device=DEVICE)
    

print(f'Validation ACC: {valid_acc:.2f}%')
print(f'Test ACC: {test_acc:.2f}%')