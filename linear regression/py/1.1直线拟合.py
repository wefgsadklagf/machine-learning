#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# 本实验用于检测直线拟合的验证
# 在拟合方法发中采用scipy.optimize 的 leastsq 方法
# * 试验较简单， 建议完全理解吸收
# * 注意数据的可视化

# In[1]:


import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# In[2]:


X = np.linspace(0, 10, 20)
Y = 3 * X + 1 + np.random.normal(scale=1.5, size = X.shape)


# In[3]:


plt.plot(X, Y, 'bo', label='Data with noise')
# 给图像添加图例
plt.legend()


# 定义直线的函数

# In[4]:


def func(W, x):
    k, b = W
    return k * x + b


# 定义误差函数

# In[5]:


def error(W, x, y, step):
    step[0] = step[0] + 1
    print("Iteration:" , step[0])
    return func(W, x) - y


# In[6]:


W0 = [100, 2]
step = [0]
# 把error 函数中其他参数打包到args中
'''
func：误差函数
x0：表示函数的参数
args（）表示数据点
'''
lst = leastsq(error, W0, args=(X, Y, step))
k, b = lst[0]
print("K = ", k, "b = ", b)


# 拟合结果

# In[7]:


X_points = np.linspace(0, 10, 100)
Y_points = k * X_points + b
plt.plot(X, Y, 'bo', label = 'Data with noise')
plt.plot(X_points, Y_points, 'r', label = 'Fitting data')
plt.legend()


# In[ ]:




