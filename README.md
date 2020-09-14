<!-- TOC -->

- [machine-learning](#machine-learning)
	- [软件目录 soft](#软件目录-soft)
	- [书籍目录 book](#书籍目录-book)
	- [文件目录说明](#文件目录说明)
	- [项目目录如下：](#项目目录如下)
		- [1.线性回归 linear regression](#1线性回归-linear-regression)
		- [2.感知机 Perceptron](#2感知机-perceptron)
		- [3.K邻近 K-neighbor algorithm](#3k邻近-k-neighbor-algorithm)
		- [4.朴素贝叶斯 Naive Bayes](#4朴素贝叶斯-naive-bayes)
		- [5.决策树 Decision tree](#5决策树-decision-tree)
		- [6.逻辑斯蒂回归 Logistic regression](#6逻辑斯蒂回归-logistic-regression)
		- [7.支持向量机 SVM](#7支持向量机-svm)
		- [8.提升方法 AdaBoost](#8提升方法-adaboost)
		- [9.EM算法 EM](#9em算法-em)
		- [10.隐马尔可夫模型 HMM](#10隐马尔可夫模型-hmm)
		- [11.条件随机场 CRF](#11条件随机场-crf)

<!-- /TOC -->

# machine-learning
这个是本人在机器学习所作的相关实验

参考书籍是《李航的统计学习方法》
(实验代码持续整理当中，持续更新，没有上传完全)

## 软件目录 soft

- Edraw MindMaster Pro 7.2.rar 为破解软件，用于查看脑图

## 书籍目录 book

- 这个里面是知乎、博客等网站推荐的，我也参考过的书籍，

## 文件目录说明

在对应的章节下都包括`ipynb`与`py`两个文件夹，分别对应`.ipynb`版本与`.py`版本，但是本人在实验中使用的jupyter notebook, `.py`版本是自动生成的，代码的可读性，难以得到保证.(个人原因`.py`暂时没有实践整理,整理完成之后会更新.)

## 项目目录如下：

### 1.线性回归 linear regression

- 直线拟合
- 多项式拟合

### 2.感知机 Perceptron

- 感知机的基础模型
- 感知机的基础模型二次练习
- 增强感知机
 使用自己创建的点集，发展如果不是线性可分割的数据集合在分类效果上并不好，而且分类的效果并不是随着训练的次数增加而增加的

### 3.K邻近 K-neighbor algorithm

- 最邻近算法
- k邻近算法
  有BUG, 如果是重复数据会当作一个数据进行处理，暂时没时间处理，有时间之后会更正
- 通过最邻近算法实现数字识别

### 4.朴素贝叶斯 Naive Bayes

- 朴素贝叶斯模型的基础练习
  
  - 朴素贝叶斯验证
  - 朴素贝叶斯2 就是再次实现了朴素贝叶斯模型
  
- 朴素贝叶斯模型的应用

  - 新闻分类
  - 邮件分类器

### 5.决策树 Decision tree

- 决策树总结
- ID3决策树实验验证
- C4.5 决策树实验验证

### 6.逻辑斯蒂回归 Logistic regression

- 逻辑斯蒂回归
- 逻辑斯蒂实现手写识别

### 7.支持向量机 SVM

- 支持向量机 - SOM 算法
- SVM人脸识别
- SVM实现手写识别

### 8.提升方法 AdaBoost

- AdaBoost
- AdaBoost可视化
- AdaBoost实现手写识别

### 9.EM算法 EM 

- EM算法
- 高斯混合模型正弦曲线
- 高斯混合的密度估计

### 10.隐马尔可夫模型 HMM

- HMM

### 11.条件随机场 CRF

- 条件随机场
