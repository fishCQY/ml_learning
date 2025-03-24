  # @file    Simple_Linear_Regression.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/24 19:10:06
  # @version 1.0
  # @brief 
  # @note    简单线性回归

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__),'../datasets/studentscores.csv')#os.path.dirname(__file__)：返回当前脚本的目录路径，__file__：返回当前脚本的绝对路径，os.path.join()：连接路径，返回一个路径字符串
dataset = pd .read_csv(file_path)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
#print(X)
#print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state=0)

# Fitting Simple Linear Regeression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

# Predicting the Result
Y_pred = regressor.predict(X_test)

# Visualising the Training Results
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Training Set')
plt.show()

# Visualising the Test Results
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_test,regressor.predict(X_test),color = 'blue')
plt.title('Test Set')
plt.show()
