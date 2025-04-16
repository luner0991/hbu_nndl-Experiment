'''
@ Function: 实现RNN的反向传播步骤
@Author: lxy
@ date: 2024/12/3
'''
import numpy as np
import torch.nn.functional as F
import torch
from rnn_cell_forward import rnn_cell_forward
from rnn_cell_backward import rnn_cell_backward
from  rnn_forward import  rnn_forward

# 实现RNN的反向步骤
def rnn_backward(da, caches):
    """
    da -- 所有隐藏状态的上游梯度，形状为(n_a, m, T_x)。
    caches -- 包含来自前向传播（rnn_forward）信息的元组。

    返回：
    gradients -- 包含以下内容的Python字典：
                        dx -- 相对于输入数据的梯度，形状为(n_x, m, T_x)的numpy数组。
                        da0 -- 相对于初始隐藏状态的梯度，形状为(n_a, m)的numpy数组。
                        dWax -- 相对于输入权重矩阵的梯度，形状为(n_a, n_x)的numpy数组。
                        dWaa -- 相对于隐藏状态权重矩阵的梯度，形状为(n_a, n_a)的numpy数组。
                        dba -- 相对于偏置的梯度，形状为(n_a, 1)的numpy数组。
    """

    ###========================= 以下是反向传播的具体计算步骤================= ###
    # 1、从caches的第一个缓存（t = 1时）中获取值
    (caches_list, x) = caches
    (a1, a0, x1, parameters) = caches_list[0]  # 获取t = 1时的值

    # 2、从da和x1的形状中获取维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 3、用合适的大小初始化各个梯度
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    # 4、循环遍历所有时间步
    for t in reversed(range(T_x)):
        # 在时间步t计算梯度。在反向传播步骤中，要明智地选择“da_next”和“cache”来使用
        # 这里将当前时间步的梯度da[:, :, t]与上一个时间步累积的梯度da_prevt相加作为输入
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches_list[t])
        # 从计算得到的梯度字典中获取各个导数（大约一行代码）
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]
        # 通过加上在时间步t的导数来累加相对于参数的全局导数
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # 5、将经过所有时间步反向传播得到的a的梯度赋给da0
    da0 = da_prevt
    ### ===================结束反向传播的计算步骤 ====================###

    # 将计算得到的各个梯度存储在一个Python字典中
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

np.random.seed(1)
x = np.random.randn(3, 10, 4)
a0 = np.random.randn(5, 10)
Wax = np.random.randn(5, 3)
Waa = np.random.randn(5, 5)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)

# 输出相对于输入数据梯度中第二行第三列的元素值
print("相对于输入数据梯度中第二行第三列的元素值 =", gradients["dx"][1][2])
print("相对于输入数据梯度的形状 =", gradients["dx"].shape)

# 输出相对于初始隐藏状态梯度中第三行第四列的元素值
print("相对于初始隐藏状态梯度中第三行第四列的元素值 =", gradients["da0"][2][3])
print("相对于初始隐藏状态梯度的形状 =", gradients["da0"].shape)

# 输出相对于输入权重矩阵梯度中第四行第二列的元素值
print("相对于输入权重矩阵梯度中第四行第二列的元素值 =", gradients["dWax"][3][1])
print("相对于输入权重矩阵梯度的形状 =", gradients["dWax"].shape)

# 输出相对于隐藏状态权重矩阵梯度中第二行第三列的元素值
print("相对于隐藏状态权重矩阵梯度中第二行第三列的元素值 =", gradients["dWaa"][1][2])
print("相对于隐藏状态权重矩阵梯度的形状 =", gradients["dWaa"].shape)

# 输出相对于偏置梯度中第五行的元素值（这里是一个包含一个元素的列表形式）
print("相对于偏置梯度中第五行的元素值 =", gradients["dba"][4])
print("相对于偏置梯度的形状 =", gradients["dba"].shape)