  # @file    KNN.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/25 15:59:52
  # @version 1.0
  # @brief 

import numpy as np  # 导入numpy库，用于数值计算和数组操作。
import pandas as pd  # 导入pandas库，用于数据处理和分析。
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于数据可视化。
import os  # 导入os库，用于操作系统相关的功能。

file_path = os.path.join(os.path.dirname(__file__),'../datasets/Social_Network_Ads.csv')  # 构建文件路径，使用__file__获取当前脚本文件的路径，然后使用os.path.dirname获取父目录路径，最后拼接数据集文件名。这行代码的目的是获取数据集文件的完整路径。
dataset = pd.read_csv(file_path)  # 使用pandas的read_csv函数读取csv文件，将数据存储在名为dataset的DataFrame中

X=dataset.iloc[:,[2,3]].values  # 从dataset中选择第2列和第3列作为特征，将其转换为numpy数组，并赋值给变量X。这行代码的目的是获取特征数据。
Y = dataset.iloc[:,-1].values  # 从dataset中选择最后一列作为标签，将其转换为numpy数组，并赋值给变量Y。这行代码的目的是获取标签数据

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)  # 使用sklearn的train_test_split函数将特征数据X和标签数据Y随机划分为训练集和测试集。test_size参数指定测试集的比例为0.2，random_state参数用于设置随机种子，保证每次运行结果一致。这行代码的目的是划分数据集。

# feature scaling
from sklearn.preprocessing import StandardScaler  # 从sklearn.preprocessing模块导入StandardScaler类，用于特征缩放。这行代码的目的是导入特征缩放工具。
sc = StandardScaler()  # 创建StandardScaler对象，用于特征缩放。这行代码的目的是创建一个特征缩放对象。
X_train = sc.fit_transform(X_train)
X_test =sc.transform(X_test)  # 使用训练集的均值和标准差对测试集进行特征缩放。这行代码的目的是对测试集进行特征缩放。

# Fitting KNN to the training set
from sklearn.neighbors import KNeighborsClassifier
# n_neighbors：指定KNN算法中的邻居数量，默认为5。可以根据具体问题和数据集的特点调整该参数。
# metric：指定距离度量函数，默认为'minkowski'。常用的距离度量函数包括欧氏距离（'euclidean'）、曼哈顿距离（'manhattan'）和闵可夫斯基距离（'minkowski'）等。
# p：指定距离度量函数的参数，默认为2。当metric为'minkowski'时，该参数表示使用的是哪种距离度量函数，例如p=2表示使用欧氏距离，p=1表示使用曼哈顿距离。
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_test,Y_pred)
# 混淆矩阵的第一行是对负类样本的预测，第二行是对正类样本的预测，主对角线是预测正确的样本，副对角线是预测错误的样本
print(cm)  # 打印混淆矩阵，用于评估分类模型的准确性。这行代码的目的是打印混淆矩阵。
print(classification_report(Y_test,Y_pred))  # 打印分类报告，用于评估分类模型的准确性。这行代码的目的是打印分类报告。
# precision 精确率：预测为正类的样本中准确的比率
# recall (召回率)：真实正类样本中的正类准确率
# f1-score：精确率和召回率的调和平均
# support：实际样本数
# accuracy：整体准确率 (TP+TN)/Total = 93%
# macro avg：各类别指标的简单平均
# weighted avg：按样本量加权的平均