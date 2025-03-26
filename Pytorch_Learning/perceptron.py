  # @file    perceptron.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/25 21:43:51
  # @version 1.0
  # @brief 

import numpy as np
import matplotlib.pyplot as plt
import torch

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# dataset
# delimiter参数设置为'\t'，表示列之间的分隔符是制表符，也就是Tab键分隔的数据。
# genfromtxt默认会返回一个二维的NumPy数组，每一行是数据的一行，每一列是对应的特征或标签
data = np.genfromtxt('../datasets/perceptron_toydata.txt',delimiter='\t')
X,y = data[:,:2],data[:,2]
y = y.astype(np.int64)# 原始数据中的类别标签可能存储为浮点数，PyTorch的交叉熵损失函数要求标签为int64类型

# print('Class label counts:',np.bincount(y))
# print('X.shape:',X.shape)
# print('y.shape',y.shape)

#Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0]) # 生成的是包含所有样本索引的数组（假设总共有n个样本，则数组是[0,1,2,...,n-1]）
shuffle_rng = np.random.RandomState(123)# 设置了固定种子(123)的随机数生成器，后续可以使用它的shuffle()方法来打乱索引顺序
shuffle_rng.shuffle(shuffle_idx) # 打乱索引顺序
X,y = X[shuffle_idx],y[shuffle_idx]

# 前70个训练，后30个测试
X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]


# Normalize(mean zero,unit variance)
# 如果不加 axis=0，会计算整个矩阵的均值和标准差（得到一个标量值）
# 加了 axis=0 后，返回形状为 (2,) 的数组（每个特征对应一个统计量），按列进行计算
mu,sigma = X_train.mean(axis=0),X_train.std(axis=0)
X_train = (X_train-mu)/sigma
X_test = (X_test-mu)/sigma

plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],label ='class 0',marker='o')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],label ='class 1',marker='s')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()

# if have gpu ,use gpu else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def custom_where(cond, x_1, x_2):
    """自定义条件选择函数,模拟torch.where的功能
    参数:
        cond (Tensor): 布尔掩码张量(0/1值)
        x_1 (Tensor): 条件为真时选取的值
        x_2 (Tensor): 条件为假时选取的值
    返回:
        Tensor: 合并后的新张量，数学表达式为 cond*x_1 + (1-cond)*x_2
    """
    # 通过矩阵运算实现条件选择：
    # 当cond=1时保留x_1对应位置的值，x_2部分归零
    # 当cond=0时保留x_2对应位置的值，x_1部分归零
    # 使用逻辑非操作代替 1 - cond
    return (cond * x_1) + (~cond * x_2)
# define the perceptron model，手动实现
class Perceptron():
    def __init__(self,num_features):
        self.num_features = num_features
        # 初始化权重矩阵 [num_features, 1] 与输入特征进行矩阵乘法，使用预定义的device
        self.weights = torch.zeros(num_features,1,dtype=torch.float32,device=device)
        # 初始化偏置项，形状为[1]的标量
        self.bias = torch.zeros(1,dtype = torch.float32,device = device)

    def forward(self,x):
        """前向传播计算预测值
        参数:
            x (Tensor): 输入特征张量，形状应为 [batch_size, num_features]
        返回:
            Tensor: 二分类预测结果 (0或1)，形状 [batch_size, 1]
        """
        linear = torch.add(torch.mm(x,self.weights),self.bias)# 等价于数学运算 x · w+b
        predictions = custom_where(linear>0.,1,0).float()  # 通过自定义where函数应用阶跃激活函数
        return predictions
    
    def backward(self, x, y):
        """计算预测误差
        参数:
            y (Tensor): 真实标签，形状 [batch_size, 1]
        返回:
            Tensor: 误差信号，形状 [batch_size, 1]
        """
        predictions = self.forward(x)
        errors = y - predictions  # 直接计算误差用于参数更新
        return errors
    
    def train(self,x,y,epochs):
        """训练感知机模型
        参数:
            x (Tensor): 训练特征张量 [n_samples, n_features]
            y (Tensor): 训练标签张量 [n_samples]
            epochs (int): 训练迭代次数
        """
        # 外层循环控制训练轮次
        for e in range(epochs):
            # 内层循环逐个样本更新参数
            for i in range(y.size()[0]):
                # 计算当前样本的误差（y_pred - y_true）
                errors = self.backward(x[i].view(1,self.num_features),y[i]).view(-1)
                # 更新权重：w = w + error * x_i，线性模型w的梯度为x，这里可视作学习率为1的随机梯度下降
                self.weights +=(errors*x[i]).view(self.num_features,1)
                # 更新偏置：b = b + error
                self.bias +=errors
    
    def evaluate(self,x,y):
        """评估模型准确率
        参数:
            x (Tensor): 测试特征张量 [n_samples, n_features]
            y (Tensor): 测试标签张量 [n_samples]
        返回:
            float: 模型在测试集上的准确率
        """
        # 获取预测结果并计算正确率
        predictions = self.forward(x).view(-1)
        accuracy = torch.sum(predictions==y).float()/y.size()[0]
        return accuracy
    
# training the perceptron
ppn = Perceptron(num_features=2)
# 将NumPy数组转换为PyTorch张量
# X_train: 预处理后的训练数据（NumPy数组）
# dtype=torch.float32: 指定张量数据类型为32位浮点数（与模型参数类型匹配）
# device=device: 将张量移动到之前定义的设备（GPU/CPU）上，确保与模型参数在同一设备
X_train_tensor = torch.tensor(X_train,dtype=torch.float32,device=device) 
y_train_tensor = torch.tensor(y_train,dtype=torch.float32,device=device) 

ppn.train(X_train_tensor,y_train_tensor,epochs=5)

print('Model parameters')
print('weights: %s' % ppn.weights)
print('Bias: %s' % ppn.bias)

# evaluating the model
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = ppn.evaluate(X_test_tensor,y_test_tensor)
print('Test set accurracy: %.2f%%' %(test_acc*100))

##########################
### 2D Decision Boundary
##########################

# 获取训练好的模型参数（权重向量w和偏置b）
# w是形状为[2,1]的张量，对应两个特征的权重
# b是标量张量，表示偏置项
# 获取模型参数（确保张量在CPU上）
w, b = ppn.weights.cpu(), ppn.bias.cpu()

# 计算决策边界的y值（转换为Python原生类型）
x_min = -2
y_min = ((-(w[0].item() * x_min) - b.item()) / w[1].item())

x_max = 2
y_max = ((-(w[0].item() * x_max) - b.item()) / w[1].item())

fig,ax = plt.subplots(1,2,sharex=True,figsize=(7,3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])
ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()