import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

import torch
import torch.nn.functional as F

#prepare the dataset
# 创建数据源对象用于读取远程文件
ds = np.lib.DataSource()
# 打开鸢尾花数据集URL（实际会获得副本文件）
fp = ds.open('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')


# fp.read()：读取网络文件对象中的原始字节数据（bytes类型）,encode进行类型转换str
# BytesIO()：将字节数据包装成类似文件的内存对象，供genfromtxt直接读取
# 整个过程在内存中完成，避免写入物理磁盘
x = np.genfromtxt(BytesIO(fp.read().encode()), delimiter=',', usecols=range(2), max_rows=100)

# 创建标签数组：前50个样本（Iris-setosa）标记为0，后50个（Iris-versicolor）标记为1
y = np.zeros(100)
y[50:] = 1  # 第50-99索引位置赋值为1

np.random.seed(1)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
X_test,y_test = x[idx[:25]],y[idx[:25]]
X_train,y_train = x[idx[25:]],y[idx[25:]]
mu,std = np.mean(X_train,axis=0),np.std(X_train,axis=0)
X_train,X_test = (X_train -mu)/std,(X_test-mu)/std

# 画散点图观察原始数据的数据分布
fig,ax = plt.subplots(1,2,figsize=(7,2.5))
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
plt.show()

# low_level implementation  with manual gradients
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def custom_where(cond,x_1,x_2):
    return (cond*x_1)+(~cond*x_2)

class LogisticRegression1():
    def __init__(self,num_features):
        self.num_features = num_features
        # 避免影响计算速度，一般用float32，默认是用float64,所以需要修改
        self.weights = torch.zeros(num_features,1,dtype=torch.float32,device=device)
        self.bias = torch.zeros(1,dtype=torch.float32,device=device)

# Sigmoid激活函数，将线性输出映射到(0,1)概率区间
    def _sigmoid(self,z):
        return 1./(1.+torch.exp(-z))
    def forward(self,x):
        linear = torch.add(torch.mm(x,self.weights),self.bias)
        probas = self._sigmoid(linear)
        return probas
    def backward(self,probas,y):
        errors = y - probas.view(-1) # 真实值减去与预测值，view(-1)转为列向量
        return errors
    #计算二元交叉熵损失（Binary Cross-Entropy Loss）
    # -(mean(y*log(p) + (1-y)*log(1-p)))
    def _logit_cost(self,y,proba):
        tmp1 = torch.mm(-y.view(1,-1),torch.log(proba))# y.view(1, -1) 转换为行向量（形状 [1, batch_size]）
        tmp2 = torch.mm((1-y).view(1,-1),torch.log(1-proba))
        return tmp1-tmp2
    def train(self,x,y,num_epochs,learning_rate=0.01):
        for e in range(num_epochs):
            # compute outputs
            probas = self.forward(x)
            # compute gradients
            errors = self.backward(probas,y)
            # 二项logistic分布使用极大似然估计得到的就是权重梯度=X_T⋅(y-probas)(形状 [num_features,1])
            neg_grad = torch.mm(x.transpose(0,1),errors.view(-1,1))
            #update weights
            self.weights+=learning_rate*neg_grad
            self.bias+=learning_rate*torch.sum(errors)
            #logging
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % self._logit_cost(y, self.forward(x)))

    def predict_labels(self,x):
        probas= self.forward(x)
        labels = custom_where(probas>.5,1,0)
        return labels
    def evaluate(self,x,y):
        labels = self.predict_labels(x).view(-1)
        accuracy = torch.sum(labels == y).float() / y.size()[0]
        return accuracy

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
logr=LogisticRegression1(num_features=2)
logr.train(X_train_tensor,y_train_tensor,num_epochs=10,learning_rate=0.1)
print('\nModel parameters:')
print('  Weights: %s' % logr.weights)
print('  Bias: %s' % logr.bias)

#Evaluating the model
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = logr.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))

##########################
### 2D Decision Boundary
##########################

w, b = logr.weights, logr.bias
#.item() 的作用是将 ​单元素PyTorch张量 转换为 ​Python原生数值​（如 float 或 int）
x_min = -2
y_min = ( (-(w[0].item() * x_min) - b[0].item())
          / w[1].item() )

x_max = 2
y_max = ( (-(w[0].item() * x_max) - b[0].item())
          / w[1].item() )


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])

ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()

# low_level implementation using autograd
class LogisticRegression2():
    def __init__(self,num_features):
        self.num_features = num_features
        # 避免影响计算速度，一般用float32，默认是用float64,所以需要修改,使用自动梯度求导
        self.weights = torch.zeros(num_features,1,dtype=torch.float32,device=device,requires_grad=True)
        self.bias = torch.zeros(1,dtype=torch.float32,device=device,requires_grad=True)

# Sigmoid激活函数，将线性输出映射到(0,1)概率区间
    def _sigmoid(self,z):
        return 1./(1.+torch.exp(-z))
    def forward(self,x):
        linear = torch.add(torch.mm(x,self.weights),self.bias)
        probas = self._sigmoid(linear)
        return probas
    def backward(self,probas,y):
        errors = y - probas.view(-1) # 真实值减去与预测值，view(-1)转为列向量
        return errors
    #计算二元交叉熵损失（Binary Cross-Entropy Loss）
    # -(mean(y*log(p) + (1-y)*log(1-p)))
    def _logit_cost(self,y,proba):
        tmp1 = torch.mm(-y.view(1,-1),torch.log(proba))# y.view(1, -1) 转换为行向量（形状 [1, batch_size]）
        tmp2 = torch.mm((1-y).view(1,-1),torch.log(1-proba))
        return tmp1-tmp2
    def train(self,x,y,num_epochs,learning_rate=0.01):
        for e in range(num_epochs):
            # compute outputs
            proba = self.forward(x)
            cost = self._logit_cost(y,proba)
            # compute gradients
            cost.backward() #这里使用损失函数进行反向传播，参数是在模型初始化时就确定了
            #update weights
            #detach() 创建了一个与计算图断开的新张量 tmp，对 tmp 的修改不会影响原参数 self.weights
            #相当于对参数进行拷贝，python里对变量赋值相当于对c++里的引用
            tmp = self.weights.detach()
            tmp -= learning_rate*self.weights.grad

            tmp = self.bias.detach()
            tmp -= learning_rate* self.bias.grad

            #Reset gradients to zero for next iteration
            # PyTorch梯度是累加的，需手动清零以准备下一次迭代。
            self.weights.grad.zero_()
            self.bias.grad.zero_()

            #logging
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % self._logit_cost(y, self.forward(x)))

    def predict_labels(self,x):
        probas= self.forward(x)
        labels = custom_where(probas>.5,1,0)
        return labels
    def evaluate(self,x,y):
        labels = self.predict_labels(x).view(-1)
        accuracy = torch.sum(labels == y).float() / y.size()[0]
        return accuracy
    
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

logr = LogisticRegression2(num_features=2)
logr.train(X_train_tensor, y_train_tensor, num_epochs=10, learning_rate=0.1)

print('\nModel parameters:')
print('  Weights: %s' % logr.weights)
print('  Bias: %s' % logr.bias)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = logr.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))

##########################
### 2D Decision Boundary
##########################

w, b = logr.weights, logr.bias
#.item() 的作用是将 ​单元素PyTorch张量 转换为 ​Python原生数值​（如 float 或 int）
x_min = -2
y_min = ( (-(w[0].item() * x_min) - b[0].item())
          / w[1].item() )

x_max = 2
y_max = ( (-(w[0].item() * x_max) - b[0].item())
          / w[1].item() )


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])

ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()

#high-level implentation using the nn.model API
class LogisticRegression3(torch.nn.Module):
    def __init__(self,num_features):
        """使用PyTorch高层API构建逻辑回归模型
        参数:
            num_features (int): 输入特征维度
        功能:
            1. 继承nn.Module基类
            2. 创建线性层 (w^T x + b)
            3. 显式初始化参数为全零
        """
        #所有自定义PyTorch模型类必须继承 nn.Module 并调用 super().__init__()，否则模型无法正常工作
        super(LogisticRegression3,self).__init__()  # 必须的父类初始化
        #num_features：输入特征的数量（如特征数为2，则输入形状为 [batch_size, 2]）。
        #1：输出特征的维度（逻辑回归输出一个标量，表示概率）
        self.linear = torch.nn.Linear(num_features,1)  # 创建线性层 (包含weight和bias)
        
        # 默认情况下，nn.Linear 的权重和偏置会随机初始化。如果需要手动初始化（如置零）
        self.linear.weight.detach().zero_()  # 权重矩阵 [num_features, 1]
        self.linear.bias.detach().zero_()    # 偏置项 [1]

    def forward(self,x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas

model = LogisticRegression3(num_features=2).to(device)

# define cost function and set up optimizer
cost_fn = torch.nn.BCELoss(reduction='sum')# 二元交叉熵损失，将每个样本的损失相加，返回总损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)# 设置优化器（随机梯度下降）​

def comp_accuracy(label_var,pred_probas):
    pred_labels = custom_where((pred_probas>0.5),1,0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float()/label_var.size(0)
    return acc
num_epochs =10
X_train_tensor= torch.tensor(X_train,dtype=torch.float32,device=device)
y_train_tensor = torch.tensor(y_train,dtype=torch.float32,device=device).view(-1,1)

for epoch in range(num_epochs):
    # Compute outputs
    out = model(X_train_tensor)# 调用模型得到输出
    
    # Compute gradients
    cost = cost_fn(out,y_train_tensor)# 计算损失函数
    optimizer.zero_grad()# 梯度清零
    cost.backward()# 反向传播，计算梯度

    # Update weights
    optimizer.step()# 参数优化

    # Logging
    pred_probas = model(X_train_tensor)
    acc = comp_accuracy(y_train_tensor,pred_probas)
    print('Epoch: %03d' % (epoch+1),end="")
    print(' | Train ACC: %.3f' % acc, end="")
    print(' | Cost: %.3f' % cost_fn(pred_probas, y_train_tensor))

print('\nModel parameters:')
print('Weights: %s' % model.linear.weight)
print('Bias: %s' % model.linear.bias)


X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

pred_probas = model(X_test_tensor)
test_acc = comp_accuracy(y_test_tensor, pred_probas)

print('Test set accuracy: %.2f%%' % (test_acc*100))

##########################
### 2D Decision Boundary
##########################

w, b = logr.weights, logr.bias
#.item() 的作用是将 ​单元素PyTorch张量 转换为 ​Python原生数值​（如 float 或 int）
x_min = -2
y_min = ( (-(w[0].item() * x_min) - b[0].item())
          / w[1].item() )

x_max = 2
y_max = ( (-(w[0].item() * x_max) - b[0].item())
          / w[1].item() )


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])

ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()