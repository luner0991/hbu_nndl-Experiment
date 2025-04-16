'''
@ Function: 实现RNN的前向传播步骤
@Author: lxy
@ date: 2024/12/3
'''
import numpy as np
import torch.nn.functional as F
import torch
from  rnn_cell_forward import  rnn_cell_forward

def rnn_forward(x, a0, parameters):
    """
    x -- 每个时间步的输入数据，形状为(n_x, m, T_x)。
    a0 -- 初始隐藏状态，形状为(n_a, m)。
    parameters -- 包含以下内容的Python字典：
                        Waa -- 用于乘以隐藏状态的权重矩阵，形状为(n_a, n_a)的numpy数组。
                        Wax -- 用于乘以输入的权重矩阵，形状为(n_a, n_x)的numpy数组。
                        Wya -- 将隐藏状态与输出相关联的权重矩阵，形状为(n_y, n_a)的numpy数组。
                        ba -- 偏置项，形状为(n_a, 1)的numpy数组。
                        by -- 将隐藏状态与输出相关联的偏置项，形状为(n_y, 1)的numpy数组。

    返回：
    a -- 每个时间步的隐藏状态，形状为(n_a, m, T_x)的numpy数组。
    y_pred -- 每个时间步的预测结果，形状为(n_y, m, T_x)的numpy数组。
    caches -- 用于反向传播所需值的元组，包含（缓存列表，输入数据x）。
    """
    # 初始化“caches”，它将包含所有缓存的列表
    caches = []
    # 从输入数据x和权重矩阵Wya的形状中获取维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    ### ===============以下是前向传播的具体计算步骤 ===============###

    # 1、用零初始化“a”和“y”
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    #2、 初始化下一个隐藏状态
    a_next = a0

    # 3、遍历所有时间步
    for t in range(T_x):
        # 调用cell前向传播函数更新下一个隐藏状态，计算预测结果，获取缓存（大约一行代码）
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # 将新的“下一个”隐藏状态的值保存到a中（大约一行代码）
        a[:, :, t] = a_next
        # 将预测结果的值保存到y_pred中（大约一行代码）
        y_pred[:, :, t] = yt_pred
        # 将“cache”添加到“caches”列表中（大约一行代码）
        caches.append(cache)

    ### ====================结束前向传播的计算步骤============= ###

    # 将反向传播所需的值存储在缓存中
    caches = (caches, x)

    return a, y_pred, caches

if __name__=="main":
    np.random.seed(1)
    x = np.random.randn(3, 10, 4)
    a0 = np.random.randn(5, 10)
    Waa = np.random.randn(5, 5)
    Wax = np.random.randn(5, 3)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    a, y_pred, caches = rnn_forward(x, a0, parameters)

    # 输出隐藏状态数组中第五行（索引从0开始，所以索引4对应第五行）第二列（索引从0开始，所以索引1对应第二列）的元素值
    print("隐藏状态数组中第五行第二列的元素值 = ", a[4][1])
    print("隐藏状态数组的形状 = ", a.shape)

    # 输出预测结果数组中第二行（索引从0开始，所以索引1对应第二行）第四列（索引从0开始，所以索引3对应第四列）的元素值
    print("预测结果数组中第二行第四列的元素值 =", y_pred[1][3])
    print("预测结果数组的形状 = ", y_pred.shape)

    # 输出缓存元组中第二个元素（索引从0开始，所以索引1对应第二个元素）的列表中第四个元素（索引从0开始，所以索引3对应第四个元素）的值
    print("缓存元组中第二个元素的列表中第四个元素的值 =", caches[1][1][3])
    print("缓存元组的长度 = ", len(caches))