# MLA
Machine Learning Algorithm

## 安装
安装numpy
参考：[ python实践系列之（一）安装 python/pip/numpy/matplotlib](http://blog.csdn.net/sinat_28224453/article/details/51462935)

运行 gmm.py

## 高斯混合模型(GMM)

1. 高斯混合模型的定义:高斯混合模型是指具有如下形式的概率分布模型：

P(Y|θ)=∑k=1Kαkϕ(y|θk)
其中αk是系数，αk ≥ 0,∑k = 1Kαk=1;ϕ(y|θk)是高斯分布(正态分布)密度函数，θk=(μk,σk2),
ϕ(y|θk)=1(2π)σkexp(−(y−μk)22σk2) 称为第k个分模型。

2. 使用EM算法估计高斯混合模型

具体推导过程从略，可以参见《统计学习方法》。这里直接给出结果：

高斯混合模型的EM估计算法

输入：观测数据y1,y2,...,yN，高斯混合模型；
输出：高斯混合模型的参数。

(1)取参数的初始值迭代
(2)E步，依据当前模型的参数，计算分模型k对观测数据yj的响应度，γ^jk=αkϕ(yj|θk)∑k=1Kαkϕ(yj|θk),j=1,2,...,N,k=1,2,...K
(3)计算新一轮迭代的模型参数

μ^k=∑j=1Nγ^jkyj∑j=1Nγ^jk,k=1,2,...,K

σ^k2=∑j=1Nγ^jk(yj−μk)2∑j=1Nγ^jk,k=1,2,...,K

α^k=∑j=1Nγ^jkN,k=1,2,...,K

(4)重复(2)和(3)直到收敛。

## NMI 
实现参考：https://smj2284672469.github.io/2017/11/14/NMI-Python/

## 数据集
### iris data set
安德森鸢尾花卉数据集（英文：Anderson's Iris data set），也称鸢尾花卉数据集（英文：Iris flower data set）或费雪鸢尾花卉数据集（英文：Fisher's Iris data set），是一类多重变量分析的数据集。
它最初是埃德加·安德森从加拿大加斯帕半岛上的鸢尾属花朵中提取的地理变异数据[1]，后由罗纳德·费雪作为判别分析的一个例子[2]，运用到统计学中。
其数据集包含了150个样本，都属于鸢尾属下的三个亚属，分别是山鸢尾、变色鸢尾和维吉尼亚鸢尾。四个特征被用作样本的定量分析，
它们分别是花萼和花瓣的长度和宽度。基于这四个特征的集合，费雪发展了一个线性判别分析以确定其属种。

### point data
每个点包含横坐标和竖坐标，以及所属类别

### study data
Attribute Information:

STG (The degree of study time for goal object materails), (input value);

SCG (The degree of repetition number of user for goal object materails) (input value) 

STR (The degree of study time of user for related objects with goal object) (input value) 

LPR (The exam performance of user for related objects with goal object) (input value) 

PEG (The exam performance of user for goal objects) (input value) 

UNS (The knowledge level of user) (target value) 

Very Low: 50 

Low:129 

Middle: 122 

High 130


### wholesale data
Attribute Information:

1)	FRESH: annual spending (m.u.) on fresh products (Continuous); 
2)	MILK: annual spending (m.u.) on milk products (Continuous); 
3)	GROCERY: annual spending (m.u.)on grocery products (Continuous); 
4)	FROZEN: annual spending (m.u.)on frozen products (Continuous) 
5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous) 
6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous); 
7)	CHANNEL: customersâ€™ Channel - Horeca (Hotel/Restaurant/CafÃ©) or Retail channel (Nominal) 
8)	REGION: customersâ€™ Region â€“ Lisnon, Oporto or Other (Nominal) 
Descriptive Statistics: 
