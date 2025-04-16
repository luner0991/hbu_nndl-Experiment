'''
@ Function: 实现RNN单元的前向传播步骤
@Author: lxy
@ date: 2024/12/3
'''
import numpy as np
import torch.nn.functional as F
import torch

# 实现了RNN单元的单个前向步骤
def rnn_cell_forward(xt, a_prev, parameters):
    '''
    xt：在时间步“t”的输入数据，形状为(n_x, m)的numpy数组
    a_prev：在时间步“t - 1”的隐藏状态，形状为(n_a, m)的numpy数组
    parameters：
         Wax：用于乘以输入的权重矩阵，形状为(n_a, n_x)的numpy数组
         Waa：用于乘以隐藏状态的权重矩阵，形状为(n_a, n_a)的numpy数组
         Wya：将隐藏状态与输出相关联的权重矩阵，形状为(n_y, n_a)的numpy数组
         ba：偏置项，形状为(n_a, 1)的numpy数组
         by：将隐藏状态与输出相关联的偏置项，形状为(n_y, 1)的numpy数组
    '''
    # 从parameters字典中获取各个参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算下一个激活状态
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # 计算当前单元的输出
    yt_pred = F.softmax(torch.from_numpy(np.dot(Wya, a_next) + by), dim=0)
    # 将反向传播所需的值存储在cache中
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache  # a_next：下一个隐藏状态，形状为(n_a, m) ， yt_pred：在时间步“t”的预测结果，形状为(n_y, m)的numpy数组

if __name__ =='main':
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    Wax = np.random.randn(5, 3)
    Waa = np.random.randn(5, 5)
    Wya = np.random.randn(2, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    print("下一个隐藏状态的第五行内容 = ", a_next[4])
    print("下一个隐藏状态的形状 = ", a_next.shape)
    print("在时间步“t”的预测结果的第二行内容 =", yt_pred[1])
    print("在时间步“t”的预测结果的形状 = ", yt_pred.shape)