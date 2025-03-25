  # @file    Logistic_Regression.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/25 09:25:09
  # @version 1.0
  # @brief 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__),'../datasets/Social_Network_Ads.csv')
dataset = pd.read_csv(file_path)
X= dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,-1].values

# Splitting the dateset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)# 计算训练集的均值(μ)和标准差(σ)
X_test = sc.transform(X_test)# 使用训练集计算的μ和σ来标准化测试集

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
print(classification_report(Y_test,Y_pred))# 打印分类报告，包括每个类别的精确率、召回率、F1值和支持度等指标，以及整体的精确率、召回率和F1值等指标，用于评估模型的分类性能

#Visualization
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train,Y_train
# 创建网格坐标矩阵（覆盖所有数据点范围），start和stop是基于特征的最小值和最大值扩展了1，步长是0.01
x1,x2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
## 绘制决策边界，根据预测结果的概率或类别值来填充颜色
#x1.ravel() 和 x2.ravel() 将网格矩阵展平为1D数组
# np.array([x1.ravel(),x2.ravel()]).T 生成符合模型输入的格式：[[x1,x2], [x1,x2], ...]，转置前按行堆叠[[1,2,3...],[4,5,6...]]
# reshape(x1.shape) 将预测结果恢复为2D网格结构，与原始坐标矩阵匹配
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap = ListedColormap(('red','green')))
# 设置坐标轴范围
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
# 绘制数据点散点图
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
for i,j in enumerate(np.unique(Y_test)):# 这里的j是标签值，只有两个类别，所以当j=0时，i=0,j=1时，i=1
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c = ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

X_set,Y_set = X_test,Y_test
x1,x2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(Y_test)):
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c = ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()