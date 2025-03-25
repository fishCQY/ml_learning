  # @file    Multiple_Linear_Regression.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/24 19:28:32
  # @version 1.0
  # @brief 
 
import numpy as np
import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__),'../datasets/50_Startups.csv')#os.path.dirname(__file__)：返回当前脚本的目录路径，__file__：返回当前脚本的绝对路径，os.path.join()：连接路径，返回一个路径字符串
dataset = pd.read_csv(file_path)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values #从第0数起

#State需要进行向量编码
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
# print(X)

# onehotencoder = OneHotEncoder(categorical_features = [3])#sklearn 的版本要为旧版本，categorical_feature = [3]：指定要进行独热编码的列索引，左闭右开区间
ct = ColumnTransformer([("encoder",OneHotEncoder(),[3])],remainder='passthrough')# 需要传一个tuple进去[()]
X = ct.fit_transform(X)
#print(X)

# Avoiding Dummy Variable Trap 避免虚拟变量陷阱
# 多重共线性指的是在回归模型中，两个或多个预测变量高度相关，导致模型难以估计每个变量的独立影响。
# 而完全多重共线性则是指一个变量可以完全由其他变量线性组合表示，这在回归分析中会导致矩阵不可逆，无法计算参数估计值
# 当同时存在以下两个条件时会产生完全多重共线性，所有虚拟变量之和等于1（全包含性），模型中存在截距项（常数项）
# 删除第一列后：保留N-1个虚拟变量，消除与截距项的完全共线性，保持所有必要信息（被删除列的状态可通过剩余列的0值推断）
X = X[:,1:]
# print(X)

# Splitting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
 
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

# regression evaluation
from sklearn.metrics import r2_score # R²，也就是决定系数，是用来衡量回归模型对数据的拟合程度的
# R²=1 - (SS_res / SS_tot)，其中SS_res是残差平方和，即预测值与实际值之差的平方和；SS_tot是总平方和，即实际值与均值之差的平方和
print(r2_score(Y_test,Y_pred))# 越接近1，模型拟合效果越好，越接近0，模型拟合效果越差

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(regressor.predict(X_train),Y_train,color='red',alpha=0.5)# 设置透明度(alpha)增强重叠点的可视性
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'k--',lw=2) 
plt.title('Training set')
plt.xlabel('Predicted values')
plt.ylabel('Actual values')

plt.subplot(1,2,2)
plt.scatter(Y_pred,Y_test,color='red',alpha=0.5)# 设置透明度(alpha)增强重叠点的可视性
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'k--',lw=2) # 'k--'表示黑色虚线，lw=2表示线宽为2
plt.title('Test set')
plt.xlabel('Predicted values')
plt.ylabel('Actual values')

plt.tight_layout()
plt.show()

# 残差图
plt.figure(figsize=(8,6))
residuals = Y_test -Y_pred
plt.scatter(Y_pred,residuals,color='green')
plt.axhline(y=0,color = 'k',linestyle='--')# 绘制一条水平线，y=0表示水平线的位置，color='k'表示颜色为黑色，linestyle='--'表示虚线样式。​
plt.title('Residuals Distribution')# 标题​
plt.xlabel('Predicted values')# x轴标签​
plt.ylabel('Residuals')# y轴标签​
plt.show()