'''
@ Function: 实现RNN单元的反向传播步骤
@Author: lxy
@ date: 2024/12/3
'''
import numpy as np
import torch.nn.functional as F
import torch
from  rnn_cell_forward import rnn_cell_forward


def rnn_cell_backward(da_next, cache):
    """
    da_next -- 相对于下一个隐藏状态的损失梯度
    cache -- 包含有用值的Python字典（rnn_cell_forward()的输出）
    返回gradients -- 包含以下内容的Python字典：
                        dx -- 输入数据的梯度，形状为(n_x, m)
                        da_prev -- 先前隐藏状态的梯度，形状为(n_a, m)
                        dWax -- 输入到隐藏层权重的梯度，形状为(n_a, n_x)
                        dWaa -- 隐藏层到隐藏层权重的梯度，形状为(n_a, n_a)
                        dba -- 偏置向量的梯度，形状为(n_a, 1)
    """

    # 从缓存中获取值
    (a_next, a_prev, xt, parameters) = cache

    # 从参数字典中获取值
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    ### ================以下是具体的反向传播计算步骤 =======================###
    # 1、计算tanh函数关于a_next的梯度
    # 按元素逐个计算的，即与da_next对应元素相乘，dtanh可看作中间结果的一种表示方式
    dtanh = (1 - a_next * a_next) * da_next

    #2、 计算相对于Wax的损失梯度
        # 先计算dxt，它等于da_next与Wax的转置矩阵做点积dtanh
    dxt = np.dot(Wax.T, dtanh)
       # 再计算dWax，它等于dtanh与xt的转置矩阵做点积
    dWax = np.dot(dtanh, xt.T)
    '''
    解释一下上面的公式推导：
    根据公式原理，dxt =  da_next.(  Wax.T . (1- tanh(a_next)**2) )，这里的.表示矩阵点积，
    进一步展开就是da_next.(  Wax.T . dtanh * (1/d_a_next) )，化简后得到Wax.T . dtanh
    同理，dWax =  da_next.( (1- tanh(a_next)**2). xt.T)，展开并化简后得到da_next.(  dtanh * (1/d_a_next). xt.T )，即dtanh. xt.T
    '''

    # 3、计算相对于Waa的梯度
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # 4、计算相对于偏置b的梯度
    # axis=0表示在列方向上进行操作，axis=1表示在行方向上进行操作，keepdims=True用于保持矩阵的二维特性
    dba = np.sum(dtanh, keepdims=True, axis=-1)

    ### ======================结束反向传播计算步骤======================= ###

    # 将计算得到的各个梯度存储在一个Python字典中
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

if __name__ =='main':
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    Wax = np.random.randn(5, 3)
    Waa = np.random.randn(5, 5)
    Wya = np.random.randn(2, 5)
    b = np.random.randn(5, 1)
    by = np.random.randn(2, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": b, "by": by}

    a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

    da_next = np.random.randn(5, 10)
    gradients = rnn_cell_backward(da_next, cache)

    # 输出输入数据梯度中第二行第三列的元素值
    print("输入数据梯度中第二行第三列的元素值 =", gradients["dxt"][1][2])
    print("输入数据梯度的形状 =", gradients["dxt"].shape)

    # 输出先前隐藏状态梯度中第三行第四列的元素值
    print("先前隐藏状态梯度中第三行第四列的元素值 =", gradients["da_prev"][2][3])
    print("先前隐藏状态梯度的形状 =", gradients["da_prev"].shape)

    # 输出输入到隐藏层权重梯度中第四行第二列的元素值
    print("输入到隐藏层权重梯度中第四行第二列的元素值 =", gradients["dWax"][3][1])
    print("输入到隐藏层权重梯度的形状 =", gradients["dWax"].shape)

    # 输出隐藏层到隐藏层权重梯度中第二行第三列的元素值
    print("隐藏层到隐藏层权重梯度中第二行第三列的元素值 =", gradients["dWaa"][1][2])
    print("隐藏层到隐藏层权重梯度的形状 =", gradients["dWaa"].shape)

    # 输出偏置向量梯度中第五行的元素值（这里是一个包含一个元素的列表形式）
    print("偏置向量梯度中第五行的元素值 =", gradients["dba"][4])
    print("偏置向量梯度的形状 =", gradients["dba"].shape)

    # 以下是对示例中具体赋值情况的展示
    gradients["dxt"][1][2] = -0.4605641030588796
    gradients["dxt"].shape = (3, 10)
    gradients["da_prev"][2][3] = 0.08429686538067724
    gradients["da_prev"].shape = (5, 10)
    gradients["dWax"][3][1] = 0.39308187392193034
    gradients["dWax"].shape = (5, 3)
    gradients["dWaa"][1][2] = -0.28483955786960663
    gradients["dWaa"].shape = (5, 5)
    gradients["dba"][4] = [0.80517166]
    gradients["dba"].shape = (5, 1)