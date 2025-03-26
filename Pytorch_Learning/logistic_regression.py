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
        errors = y - probas.view(-1)
        return errors
    def _logit_cost(self,y,proba):
        tmp1 = torch.mm(-y.view(1,-1),torch.log(proba))
        tmp2 = torch.mm((1-y).view(1,-1),torch.log(1-proba))
        return tmp1-tmp2
    def train(self,x,y,num_epochs,learning_rate=0.01):
        for e in range(num_epochs):
            # compute outputs
            probas = self.forward(x)
            # compute gradients
            errors = self.backward(probas,y)
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