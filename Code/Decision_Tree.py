  # @file    Decision_Tree.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/25 16:57:33
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

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

#Visualization
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train,Y_train

# 创建网格坐标矩阵（覆盖所有数据点范围），start和stop是基于特征的最小值和最大值扩展了1，步长是0.01
x1,x2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

# 绘制决策边界，根据预测结果的概率或类别值来填充颜色，这里使用的是分类器的predict方法，对于回归问题，可能需要使用predict_proba方法来预测概率值。然后，将预测结果映射到颜色上，形成决策边界的可视化效果。
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualization for test set
X_set,Y_set = X_test,Y_test
# 创建网格坐标矩阵（覆盖所有数据点范围），start和stop是基于特征的最小值和最大值扩展了1，步长是0.01
x1,x2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

# 绘制决策边界，根据预测结果的概率或类别值来填充颜色，这里使用的是分类器的predict方法，对于回归问题，可能需要使用predict_proba方法来预测概率值。然后，将预测结果映射到颜色上，形成决策边界的可视化效果。
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()