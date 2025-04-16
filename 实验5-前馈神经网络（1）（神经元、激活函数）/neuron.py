
import torch
import torch.nn as nn
# 3个特征数为5的样本
X = torch.rand(size=[3, 5])
# 含有5个参数的权重向量
w = torch.rand(size=[5, 1])
# 偏置项
b = torch.rand(size=[1, 1])

# 使用'torch.matmul'实现矩阵相乘
z = torch.matmul(X, w) + b
print("（一）手动计算")
print("输入 X:", X)
print("权重 w:", w, "\n偏置 b:", b)
print("输出 z:", z)
#=======================================================================
# 定义线性层，输入维度为5，输出维度为1
linear_layer = nn.Linear(in_features=5, out_features=1,bias=False)
# 使用线性层对输入X进行变换
z = linear_layer(X)
# 打印输入、权重、偏置和输出
print("(二）使用nn.Linear函数计算")
print("输入 X:", X)
print("线性层权重 w:", linear_layer.weight)
print("线性层偏置 b:", linear_layer.bias)
print("输出 z:", z)
