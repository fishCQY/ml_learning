  # @file    Data_Preprocessing.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/03/24 16:49:12
  # @version 1.0
  # @brief 

import numpy as np
import pandas as pd
import os

# # 当CWD是项目根目录时（常见于IDE）：dataset = pd.read_csv('datasets/Data.csv')
# 当CWD是脚本目录时（常见于命令行执行）：dataset = pd.read_csv('../datasets/Data.csv')
# dataset = pd.read_csv('datasets/Data.csv')# ../：返回上级目录
# 跨平台解决方案
file_path = os.path.join(os.path.dirname(__file__),'../datasets/Data.csv')#os.path.dirname(__file__)：返回当前脚本的目录路径，__file__：返回当前脚本的绝对路径，os.path.join()：连接路径，返回一个路径字符串
dataset=pd.read_csv(file_path)
X = dataset.iloc[:,:-1].values# 取除了最后一个标签的所有行，返回的是一个数组
Y = dataset.iloc[:,-1].values# 取最后一个标签的所有行，返回的是一个数组

# print(X)
# print(Y)

# 处理缺失值
from sklearn.impute import SimpleImputer #新版本使用simple imputer类
imputer =SimpleImputer(missing_values=np.nan,strategy='mean')# 实例化，missing_values=np.nan：缺失值的表示方式，strategy='mean'：缺失值的处理方式，mean：均值，median：中位数，most_frequent：众数，constant：常量，fill_value=0：常量的值
imputer = imputer.fit(X[:,1:3])# 拟合，fit()：拟合模型，transform()：转换数据，fit_transform()：拟合模型并转换数据,左闭右开区间
X[:,1:3] = imputer.transform(X[:,1:3])# 转换数据，左闭右开区间
# print(X)

# 编码数据
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer # 允许对不同的列应用不同的转换器，并将结果合并成一个新的特征矩阵，方便在Pipeline中使用
# "encoder"：转换步骤名称（建议命名以便调试）OneHotEncoder()：实际转换器对象 [0]：应用转换的列索引（第0列）
ct = ColumnTransformer([("encoder",OneHotEncoder(),[0])],remainder='passthrough')# 实例化，remainder='passthrough'（默认会丢弃未指定的列）：保留其余的列
X = ct.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
# print(X)
# print(Y)

# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)# test_size=0.2：测试集占20%，random_state=0：随机种子，保证每次运行结果相同
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # # 使用z-score标准化： (x - μ)/σ
X_train = sc_X.fit_transform(X_train)# 1. 计算训练集的均值(μ)和标准差(σ)# 2. 用这些参数标准化训练集
X_test = sc_X.transform(X_test)# 使用训练集计算的μ和σ来标准化测试集# 避免数据泄露的关键步骤
print(X_train)
print(X_test)