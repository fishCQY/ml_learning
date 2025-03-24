import pandas as pd
import numpy as np
import warnings# 忽略普通警告
warnings.filterwarnings('ignore')

vg_df = pd.read_csv('datasets/vgsales.csv', encoding='ISO-8859-1')# 文件使用的是ISO标准
# print(vg_df[['Name','Platform','Year','Genre','Publisher']].iloc[1:7])# 显示前7行数据
# print(vg_df.iloc[:,:-1].values)# 显示所有行数据，从第1列到倒数第2列

genres = np.unique(vg_df['Genre'])# 显示所有的类别,去重并按升序排列
# print(genres)

#编码标签
from sklearn.preprocessing import LabelEncoder
gle = LabelEncoder()   # 实例化
genre_labels = gle .fit_transform(vg_df['Genre'])  # 学习类型类别，建立映射表并转换为数值标签
genre_mappings = {index: label for index,label in enumerate(gle.classes_)} # enumerate(gle.classes_)，这个函数会将列表中的每个元素与其索引组成元组，生成一个可迭代对象
# print(genre_mappings)

vg_df['GenreLabel'] = genre_labels# 将转换后的数值标签添加到原数据框中
# print(vg_df[['Name','Platform','Year','Genre','GenreLabel']].iloc[1:7])# 显示前7行数据

# 使用map的方式
gen_ord_map = {label:index for index,label in enumerate(gle.classes_)}# 建立映射表，将类别转换为数值标签
# print(gen_ord_map)

vg_df['GenreMap'] = vg_df['Genre'].map(gen_ord_map) # 遍历Genre列的每个值,在字典中查找对应的数值编码,将结果存入新列GenreMap
# print(vg_df[['Name','Genre','GenreLabel','GenreMap']].iloc[1:7])# 显示前7行数据

#One-Hot编码
from sklearn.preprocessing import OneHotEncoder
gen_ohe = OneHotEncoder()   # 实例化
gen_feature_arr = gen_ohe.fit_transform(vg_df[['GenreLabel']]).toarray()    # 学习类型类别，建立映射表并转换为数值标签
# print(gen_feature_arr)

genres = np.unique(vg_df['Genre'])  # 显示所有的类别,去重并按升序排列
gen_features = pd.DataFrame(gen_feature_arr,columns=gle.classes_)# 将转换后的数值标签添加到原数据框中
# print(gen_features.head()) # 显示前5行数据

vg_df2 = vg_df[['Name','Genre']]
# print(vg_df2.head())# 默认显示前5行数据

vg_df_ohe = pd.concat([vg_df2,gen_features],axis=1)# axis=1：表示水平合并（沿列方向），注意传进去的要是一个列表
# print(vg_df_ohe.head())

# get dummy 更加实用的onehot,利用pandas即可实现onehot编码
get_dummy_features = pd.get_dummies(vg_df[['Genre']],drop_first=True)# drop_first=True：表示删除第一个类别，避免多重共线性，​避免虚拟变量陷阱：当类别数为 N 时，只需生成 N−1 列（最后一列可通过其他列推断）。
dummy_df = pd.concat([vg_df[['Name','Genre']],get_dummy_features],axis=1)
# print(dummy_df.shape)
# print(dummy_df.head())

get_dummy_features = pd.get_dummies(vg_df[['Genre']])# 一般用这个，一般默认为False
dummy_df_False = pd.concat([vg_df[['Name','Genre']],get_dummy_features],axis=1)
# print(dummy_df_False.shape)
# print(dummy_df_False.head())

# 二值特征化
vg_year_df = vg_df[['Name','Year']]
#print(vg_year_df.head())# 默认显示前5行数据

vg_year_df['Year_tow'] = np.where(vg_year_df['Year']>=2000,1,0)# 把2000年以上的归类为1，其它归类为0
#print(vg_year_df.head())

from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold=2000) # 设置阈值，大于2000为1
vg_year_df['Year'] =vg_year_df['Year'].fillna(0) # 数据中有Nan值，需要补0，否则无法二分
bn_year = bn.transform([vg_year_df['Year']])[0] # bn，需要传二维数组，因此这里转换为(1,n)获取转换的值，取第0列
vg_year_df['bn_year'] = bn_year
# print(vg_year_df.head())# 默认显示前5行数据

# 多项式特征，获得特征更高维度和互相关系的项
polynomial_df = vg_df[['NA_Sales','EU_Sales']]
# print(polynomial_df.head())

from sklearn.preprocessing import PolynomialFeatures
# degree=2表示生成二次多项式特征，包括各个特征的平方和交互项。interaction_only=False意味着不仅生成交互项，还包括各个特征的平方项。
# 如果设为True，就只生成交互项而没有平方项。include_bias=False表示不生成偏置项（即全1的列）
pf= PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
res = pf.fit_transform(polynomial_df)
# print(res)
# 第一列和第二列分别表示原先的第一列和第二列
# 第三列和第五列表示第一列和第二列分别的平方，第四列表示两者的乘积

intr_features= pd.DataFrame(res,columns=['NA_Sales','EU_Sales','NA_Sales^2','NA_Sales*EU_Sales','EU_Sales^2'])
# print(intr_features.head())# 默认显示前5行数据

# Bining 特征，一般用来处理年龄
bin_df = vg_df[['Name','Year']]
# print(bin_df.head())

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as spstats

fig, ax = plt.subplots() # 创建图形(figure)和坐标轴(axes)对象
# fig 代表整个画布窗口
# ax 代表具体的绘图区域
bin_df['Year'].hist(color='skyblue')# 绘制直方图
ax.set_title('Developer Global_Sales Hostrogram',fontsize=12)# 设置标题
ax.set_xlabel('Global_Sales',fontsize=12)
ax.set_ylabel('Frequency',fontsize=12)# 设置x轴和y轴的标签
# plt.show()

gle = LabelEncoder()
bin_df['Year_bin']=pd.cut(bin_df['Year'],9)# 将连续型变量Year分成9个等宽的区间（左开右闭区间），每个区间对应一个标签
bin_df['Year_bin'] = bin_df['Year_bin'].astype(str) # 转换为字符串类型
bin_year = gle.fit_transform(bin_df['Year_bin'] )# 利用LabelEncoder 方法变成1-9的数值
bin_df['Year_bin'] = bin_year
# print(bin_df.head())

# 对数分布，数据的分布是正态分布，如线性回归的时候误差项要满足正态分布，而当数据不满足的时候，则需要把数据变换成正态分布
df_log = vg_df[['Name','NA_Sales']]
df_log['NA_Sales_log']=np.log((1+df_log['NA_Sales']))
# print(df_log.head())

#画两张对比图
fig,ax = plt.subplots()
plt.subplot(1,2,1)# 1行2列，第一个图
df_log['NA_Sales_log'].hist(color='skyblue')
plt.subplot(1,2,2)# 1行2列，第二个图
df_log['NA_Sales'].hist(color='skyblue')
# plt.show()

# 日期相关特征
import datetime #  Python标准库，用于处理日期和时间
from dateutil.parser import parse # 自动识别日期格式并转换为datetime对象
import pytz # 用于处理时区

time_stamps =['2020-12-16 10:30:00.360000+00:00','2019-04-16 12:15:00.250000+00:00',
              '2018-10-16 08:30:00.750000+00:00','2019-01-16 23:30:00.255500+00:00']
df = pd.DataFrame(time_stamps,columns=['Time'])
# print(df.head())

ts_objs = np.array([pd.Timestamp(item) for item in np.array(df.Time)])# 将每个字符串转换为Pandas的Timestamp对象（带纳秒精度的时间类型）
# 将结果列表转换为NumPy数组
# ts_objs = pd.to_datetime(df.Time).values# 将每个字符串转换为Pandas的Timestamp对象（带纳秒精度的时间类型）
df['Ts_objs']= ts_objs
# print(ts_objs)
df['Year'] = df['Ts_objs'].apply(lambda d: d.year)# 提取年份
# 访问包含时间戳对象的列（该列数据已在前面的代码中转换为Timestamp类型），定义匿名函数，输入参数d表示时间戳对象
# 访问Timestamp对象的year属性（返回整数年份）
df['Month'] = df['Ts_objs'].apply(lambda d: d.month)# 提取月份
df['Day'] = df['Ts_objs'].apply(lambda d: d.day)# 提取日期
df['DayOfWeek'] = df['Ts_objs'].apply(lambda d: d.dayofweek)
df['WeekDayName'] = df['Ts_objs'].apply(lambda d: d.day_name())# 提取星期几的名称
# Pandas的Timestamp对象在较新版本中已经移除了weekday_name属性（v1.0.0之后）。我们可以使用更标准的day_name()方法来替代
df['DayOfYear'] = df['Ts_objs'].apply(lambda d: d.dayofyear)# 提取一年中的第几天
df['WeekOfYear'] = df['Ts_objs'].apply(lambda d: d.weekofyear)# 提取一年中的第几周
df['IsLeapYear'] = df['Ts_objs'].apply(lambda d: d.is_leap_year)# 提取是否为闰年
df['Quarter'] = df['Ts_objs'].apply(lambda d: d.quarter)# 提取季度
df[['Time','Ts_objs','Year','Month','Day','DayOfWeek','WeekDayName','DayOfYear','WeekOfYear','IsLeapYear','Quarter']]
# print(df.head())

df['Hour'] =  df['Ts_objs'].dt.hour# 提取小时
df['Minute'] =  df['Ts_objs'].dt.minute# 提取分钟
df['Second'] =  df['Ts_objs'].dt.second# 提取秒
df['Microsecond'] =  df['Ts_objs'].dt.microsecond# 提取毫秒
df['Utcoffset'] = df['Ts_objs'].apply(lambda d: d.utcoffset())  # UTC时间位移
df['Timezone'] =  df['Ts_objs'].dt.tz# 提取时区
df['TimezoneName'] =  df['Ts_objs'].dt.tz# 提取时区名称
df[['Time','Ts_objs','Hour','Minute','Second','Microsecond','Utcoffset','Timezone','TimezoneName']]
# print(df.head())

hour_bins = [-1, 6, 12, 18, 24]# 定义时间段的边界，# 区间划分规则：(-1,6], (6,12], (12,18], (18,24]
bin_names = ['Night', 'Morning', 'Afternoon', 'Evening']# 定义时间段的标签
df['TimeOfDay'] = pd.cut(df['Hour'], bins=hour_bins, labels=bin_names)# 对小时进行分组，将其映射到对应的时间段标签
df[['Time','Hour','TimeOfDay']]
# print(df.head())