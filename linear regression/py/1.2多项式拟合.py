#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


# 导入环境需要的包
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import leastsq


# In[2]:


# 定义原始数据 用于拟合
X = np.linspace(0, 1, 10)
Y = np.sin(2*np.pi*X)+np.random.normal(0, 0.1, size=X.shape)


# In[3]:


# 绘制原始样本点，用于判断使用什么函数拟合
plt.plot(X, Y, 'bo', label='Data with noise')
plt.legend()


# In[4]:


# 绘制y = sin(X) 的图像
X_points = np.linspace(0, 1, 500)
Y_points = np.sin(2*np.pi*X_points)


# In[5]:


plt.plot(X, Y, 'bo', label = 'Data with noose')
plt.plot(X_points, Y_points, 'g', label='Real data')
plt.legend()


# In[6]:


# 使用最小二乘法拟合
# 定义误差函数
def error(W, x, y):
    # 生成一个多项式， 其中按照x的幂次由高到低顺序排列
    f = np.poly1d(W)
    return f(x) - y


# In[7]:


M = 0
W = np.random.rand(M + 1)
lsq = leastsq(error, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[8]:


M = 1
W = np.random.rand(M + 1)
lsq = leastsq(error, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[9]:


M = 2
W = np.random.rand(M + 1)
lsq = leastsq(error, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[10]:


M = 3
W = np.random.rand(M + 1)
lsq = leastsq(error, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[11]:


M = 9
W = np.random.rand(M + 1)
lsq = leastsq(error, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[12]:


# 多项式模拟集中
fig =  plt.figure(figsize=(18, 12))
for i, M in enumerate([1, 2, 3, 9]):
    plt.subplot(2, 2, i + 1)
    
    # 创建模型并拟合
    W = np.random.rand(M + 1)
    lsq = leastsq(error, W, args=(X, Y))
    W = lsq[0]
  #  print("W:", W)
    
    # 绘制图形，以便直观的理解
    plt.plot(X, Y, 'bo', label = "Data with noise")
    plt.plot(X_points, Y_points, 'g', label = 'Real Data')
    plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
    plt.title("Polynomial highest degree = {}".format(M))
    plt.legend()


# In[13]:


weight = 0.001
def error_L2(W, x, y):
    f = np.poly1d(W)
    return np.append(f(x)-y, np.sqrt(0.5*weight*np.square(W)))

M = 9
W = np.random.rand(M + 1)
lsq = leastsq(error_L2, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[14]:


weight = 0.01
def error_L2(W, x, y):
    f = np.poly1d(W)
    return np.append(f(x)-y, np.sqrt(0.5*weight*np.square(W)))

M = 9
W = np.random.rand(M + 1)
lsq = leastsq(error_L2, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[15]:


weight = 0.1
def error_L2(W, x, y):
    f = np.poly1d(W)
    return np.append(f(x)-y, np.sqrt(0.5*weight*np.square(W)))

M = 9
W = np.random.rand(M + 1)
lsq = leastsq(error_L2, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[16]:


weight = 1
def error_L2(W, x, y):
    f = np.poly1d(W)
    return np.append(f(x)-y, np.sqrt(0.5*weight*np.square(W)))

M = 9
W = np.random.rand(M + 1)
lsq = leastsq(error_L2, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[17]:


weight = 10
def error_L2(W, x, y):
    f = np.poly1d(W)
    return np.append(f(x)-y, np.sqrt(0.5*weight*np.square(W)))

M = 9
W = np.random.rand(M + 1)
lsq = leastsq(error_L2, W, args=(X, Y))
W = lsq[0]
print("W:", W)
# 绘制图形，以便直观的理解
plt.plot(X, Y, 'bo', label = "Data with noise")
plt.plot(X_points, Y_points, 'g', label = 'Real Data')
plt.plot(X_points, np.poly1d(W)(X_points), 'r', label = 'Fitting data')
plt.legend()


# In[18]:


##############################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[19]:


# MyCodeing 
X_train = X.reshape(-1, 1)
Y_train = Y.reshape(-1, 1)

fig = plt.figure(figsize=(12, 8))
for i, order in enumerate([0, 1, 3, 9]):
    plt.subplot(2, 2, i+1)
    
    poly = PolynomialFeatures(order)
    X_train_poly = poly.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_poly, Y_train)
    
    plt.ylim(-2, 2)
    plt.plot(X, Y, 'bo', label="Data with noise")
    plt.plot(X_points, Y_points, 'g', label='Real data')
    plt.plot(X_points, lr.predict(poly.fit_transform(X_points.reshape(-1, 1))), 'r', label='Fitting data')


# In[20]:


# teacher Codeing 
X_TRAIN = X.reshape(-1, 1)
Y_TRAIN = Y.reshape(-1, 1)
fig = plt.figure(figsize=(12,8))
for i, order in enumerate([0, 1, 3, 9]):
    plt.subplot(2, 2, i+1)
    
    # 训练
    '''
    使用 sklearn.preprocessing.PolynomialFeatures 
    这个类可以进行特征的构造，构造的方式就是特征与特征相乘
    （自己与自己，自己与其他人），这种方式叫做使用多项式的方式。
    '''
    poly = PolynomialFeatures(order)
    X_TRAIN_POLY = poly.fit_transform(X_TRAIN)
    '''
    创建多项式的模型，并填充数据进行训练
    
    LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    fit_intercept:是否有截据，如果没有则直线过原点;
    normalize:是否将数据归一化;
    copy_X:默认为True，当为True时，X会被copied,否则X将会被覆写;
    n_jobs:默认值为1。计算时使用的核数
    '''
    lr = LinearRegression()
    lr.fit(X_TRAIN_POLY, Y_TRAIN)
    
    # 绘图相关设置
    plt.ylim(-2, 2)
    plt.plot(X, Y, 'bo', label = 'Data with noise')
    plt.plot(X_points, Y_points, 'g', label = 'Real data')
    plt.plot(X_points, lr.predict(poly.fit_transform(X_points.reshape(-1, 1))),
    'r', label = 'Fitting data')
    plt.title("M={}".format(order))
    plt.legend()


# In[21]:


# 导入正则项的函数
from sklearn.linear_model import Ridge


# In[22]:


M = 9
X_train = X.reshape(-1, 1)
Y_train = Y.reshape(-1, 1)
fig = plt.figure(figsize=(18, 12))
# 设置正则项的参数
# i 才是序号
# lamb 数据
for i, lamb in enumerate([0.001, 0.01, 0.1, 1, 10, 100]):
    # 创建画板
    plt.subplot(2, 3, i+1)
    
    #创建模型并训练
    poly = PolynomialFeatures(M)
    X_train_poly = poly.fit_transform(X_train)
    lr = Ridge(alpha=lamb/2)
    lr.fit(X_train_poly, Y_train)
    
    #绘制图形
    plt.ylim(-2, 2)
    plt.plot(X, Y, 'bo', label = 'Data with noise')
    plt.plot(X_points, Y_points, 'g', label = 'Real data')
    plt.plot(X_points, lr.predict(poly.fit_transform(X_points.reshape(-1, 1))),
    'r', label = 'Fitting data')
    plt.title("$\lambda$={}".format(lamb))
    plt.legend()
    


# In[23]:


############################

M = 9
poly = PolynomialFeatures(M)
X_TRAIN_POLY = poly.fit_transform(X_TRAIN)
alpha=5e-3 # 模型参数精度
beta=11.1 # 数据集噪声精度
I = np.eye(np.size(X_TRAIN_POLY, 1)) # 单位矩阵
S_INV = alpha * I + beta * np.matmul(X_TRAIN_POLY.T, X_TRAIN_POLY)
S = np.linalg.inv(S_INV)
T = np.matmul(X_TRAIN_POLY.T, Y_TRAIN)
X_points_POLY = poly.fit_transform(X_points.reshape(-1, 1))
# 计算均值
MEAN = beta * np.matmul(X_points_POLY, np.matmul(S, T))
# 计算方差与标准差
SIGMA2 = 1/beta + np.sum(np.matmul(np.matmul(X_points_POLY,
S), X_points_POLY.T), axis=1)/np.size(X_TRAIN, 0)#average
STD = np.sqrt(SIGMA2)
# 绘制均值曲线及 1 个标准差区域
plt.ylim(-2.5, 2.5)
plt.plot(X, Y, 'bo', label = 'Data with noise')
plt.plot(X_points, Y_points, 'g', label = 'Real data')
plt.plot(X_points, MEAN, c="r", label="Mean")
plt.fill_between(X_points, MEAN.flatten() - STD, MEAN.flatten() + STD,
color="pink", label="STD", alpha=0.5)
plt.title("M={}".format(M))
plt.legend()


# In[ ]:




