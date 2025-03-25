  # @file    SVM.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/25 16:22:59
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
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)# 计算训练集的均值(μ)和标准差(σ)
X_test = sc.transform(X_test)# 使用训练集计算的μ和σ来标准化测试集

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)# 线性核函数，适合线性可分的情况，对于非线性可分的情况，通常使用多项式核函数或高斯核函数等核函数来进行非线性转换，以提高模型的拟合能力和分类效果。
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
x1,x2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualising the Test set results
X_set, Y_set = X_test, Y_test
x1,x2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                    np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()