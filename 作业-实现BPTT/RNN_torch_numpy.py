'''
@ Function: 分别用numpy和torch实现RNN的前向和反向传播
@Author: lxy
@ date: 2024/12/3
'''
import torch
import numpy as np

# 定义RNNCell类，用于实现一个简单的循环神经网络单元的功能
class RNNCell:
    def __init__(self, weight_ih, weight_hh,
                 bias_ih, bias_hh):
        """
        参数:
        weight_ih: 输入到隐藏层的权重矩阵
        weight_hh: 隐藏层到隐藏层的权重矩阵
        bias_ih: 输入到隐藏层的偏置向量
        bias_hh: 隐藏层到隐藏层的偏置向量
        """
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

        # 用于存储输入数据的列表
        self.x_stack = []
        # 用于存储输入数据梯度的列表
        self.dx_list = []
        # 用于存储输入到隐藏层权重梯度的栈
        self.dw_ih_stack = []
        # 用于存储隐藏层到隐藏层权重梯度的栈
        self.dw_hh_stack = []
        # 用于存储输入到隐藏层偏置梯度的栈
        self.db_ih_stack = []
        # 用于存储隐藏层到隐藏层偏置梯度的栈
        self.db_hh_stack = []

        # 用于存储上一个隐藏状态的栈
        self.prev_hidden_stack = []
        # 用于存储下一个隐藏状态的栈
        self.next_hidden_stack = []

        # 临时缓存，用于存储上一步的隐藏状态梯度
        self.prev_dh = None

    def __call__(self, x, prev_hidden):
        """
        使类的实例可像函数一样被调用，实现前向传播计算下一个隐藏状态
        参数:
        x: 当前时间步的输入数据
        prev_hidden: 上一个时间步的隐藏状态

        返回:
        next_h: 当前时间步计算得到的下一个隐藏状态
        """
        # 将当前输入数据添加到输入数据列表中
        self.x_stack.append(x)

        # 计算下一个隐藏状态，通过双曲正切函数激活
        next_h = np.tanh(
            np.dot(x, self.weight_ih.T)
            + np.dot(prev_hidden, self.weight_hh.T)
            + self.bias_ih + self.bias_hh)

        # 将上一个隐藏状态添加到上一个隐藏状态栈中
        self.prev_hidden_stack.append(prev_hidden)
        # 将当前计算得到的下一个隐藏状态添加到下一个隐藏状态栈中
        self.next_hidden_stack.append(next_h)
        # 清空临时缓存，初始化为与下一个隐藏状态相同形状的零数组
        self.prev_dh = np.zeros(next_h.shape)
        return next_h

    def backward(self, dh):
        """
        实现反向传播，计算输入数据、权重和偏置的梯度
        参数:
        dh: 相对于当前隐藏状态的梯度

        返回:
        self.dx_list: 输入数据的梯度列表
        """
        # 从输入数据列表中取出最后一个输入数据
        x = self.x_stack.pop()
        # 从上一个隐藏状态栈中取出最后一个上一个隐藏状态
        prev_hidden = self.prev_hidden_stack.pop()
        # 从下一个隐藏状态栈中取出最后一个下一个隐藏状态
        next_hidden = self.next_hidden_stack.pop()

        # 计算双曲正切函数的梯度，根据链式法则，结合传入的当前隐藏状态梯度dh和之前的缓存prev_dh
        d_tanh = (dh + self.prev_dh) * (1 - next_hidden ** 2)
        # 更新临时缓存prev_dh，用于下一次反向传播计算
        self.prev_dh = np.dot(d_tanh, self.weight_hh)

        # 计算输入数据的梯度
        dx = np.dot(d_tanh, self.weight_ih)
        # 将计算得到的输入数据梯度插入到输入数据梯度列表的开头
        self.dx_list.insert(0, dx)

        # 计算输入到隐藏层权重的梯度
        dw_ih = np.dot(d_tanh.T, x)
        # 将计算得到的输入到隐藏层权重梯度添加到对应的栈中
        self.dw_ih_stack.append(dw_ih)

        # 计算隐藏层到隐藏层权重的梯度
        dw_hh = np.dot(d_tanh.T, prev_hidden)
        # 将计算得到的隐藏层到隐藏层权重梯度添加到对应的栈中
        self.dw_hh_stack.append(dw_hh)

        # 将双曲正切函数的梯度添加到输入到隐藏层偏置梯度的栈中
        self.db_ih_stack.append(d_tanh)
        # 将双曲正切函数的梯度添加到隐藏层到隐藏层偏置梯度的栈中
        self.db_hh_stack.append(d_tanh)

        return self.dx_list

if __name__ == '__main__':
    np.random.seed(123)
    torch.random.manual_seed(123)
    np.set_printoptions(precision=6, suppress=True)

    # 创建一个PyTorch的RNN实例，输入维度为4，隐藏层维度为5，数据类型为双精度浮点数
    rnn_PyTorch = torch.nn.RNN(4, 5).double()
    # 创建一个自定义的RNNCell实例，使用PyTorch的RNN实例的权重和偏置数据
    rnn_numpy = RNNCell(rnn_PyTorch.all_weights[0][0].data.numpy(),
                        rnn_PyTorch.all_weights[0][1].data.numpy(),
                        rnn_PyTorch.all_weights[0][2].data.numpy(),
                        rnn_PyTorch.all_weights[0][3].data.numpy())

    nums = 3
    # 生成随机的输入数据，形状为(nums, 3, 4)，即3个样本，每个样本有3个时间步，每个时间步输入维度为4
    x3_numpy = np.random.random((nums, 3, 4))
    # 将生成的随机输入数据转换为PyTorch的张量，设置requires_grad=True以便计算梯度
    x3_tensor = torch.tensor(x3_numpy, requires_grad=True)

    # 生成随机的初始隐藏状态，形状为(1, 3, 5)，即1个批次，3个样本，每个样本隐藏层维度为5
    h3_numpy = np.random.random((1, 3, 5))
    # 将生成的随机初始隐藏状态转换为PyTorch的张量，设置requires_grad=True以便计算梯度
    h3_tensor = torch.tensor(h3_numpy, requires_grad=True)

    # 生成随机的相对于隐藏状态的梯度，形状为(nums, 3, 5)，即3个样本，每个样本有3个时间步，每个时间步隐藏层维度为5
    dh_numpy = np.random.random((nums, 3, 5))
    # 将生成的随机相对于隐藏状态的梯度转换为PyTorch的张量，设置requires_grad=True以便计算梯度
    dh_tensor = torch.tensor(dh_numpy, requires_grad=True)

    # 使用PyTorch的RNN实例进行前向传播，计算得到新的隐藏状态
    h3_tensor = rnn_PyTorch(x3_tensor, h3_tensor)
    h_numpy_list = []

    h_numpy = h3_numpy[0]
    # 对每个样本进行自定义RNNCell的前向传播计算
    for i in range(nums):
        h_numpy = rnn_numpy(x3_numpy[i], h_numpy)
        h_numpy_list.append(h_numpy)

    # 使用PyTorch的RNN实例进行反向传播，传入相对于隐藏状态的梯度
    h3_tensor[0].backward(dh_tensor)
    # 对每个样本进行自定义RNNCell的反向传播计算，传入相对于隐藏状态的梯度
    for i in reversed(range(nums)):
        rnn_numpy.backward(dh_numpy[i])

    print("numpy方式得到的隐藏状态列表:\n", np.array(h_numpy_list))
    print("torch方式得到的隐藏状态:\n", h3_tensor[0].data.numpy())
    print("-----------------------------------------------")

    print("numpy方式计算的输入数据梯度列表:\n", np.array(rnn_numpy.dx_list))
    print("torch方式计算的输入数据梯度:\n", x3_tensor.grad.data.numpy())
    print("------------------------------------------------")

    print("numpy方式计算的输入到隐藏层权重梯度总和（按轴0求和）:\n",
          np.sum(rnn_numpy.dw_ih_stack, axis=0))
    print("torch方式计算的输入到隐藏层权重梯度:\n",
          rnn_PyTorch.all_weights[0][0].grad.data.numpy())
    print("------------------------------------------------")

    print("numpy方式计算的隐藏层到隐藏层权重梯度总和（按轴0求和）:\n",
          np.sum(rnn_numpy.dw_hh_stack, axis=0))
    print("torch方式计算的隐藏层到隐藏层权重梯度:\n",
          rnn_PyTorch.all_weights[0][1].grad.data.numpy())
    print("------------------------------------------------")

    print("numpy方式计算的输入到隐藏层偏置梯度总和（按轴0和轴1求和）:\n",
          np.sum(rnn_numpy.db_ih_stack, axis=(0, 1)))
    print("torch方式计算的输入到隐藏层偏置梯度:\n",
          rnn_PyTorch.all_weights[0][2].grad.data.numpy())
    print("-----------------------------------------------")

    print("numpy方式计算的隐藏层到隐藏层偏置梯度总和（按轴0和轴1求和）:\n",
          np.sum(rnn_numpy.db_hh_stack, axis=(0, 1)))
    print("torch方式计算的隐藏层到隐藏层偏置梯度:\n",
          rnn_PyTorch.all_weights[0][3].grad.data.numpy())