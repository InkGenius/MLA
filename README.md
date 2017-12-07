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
