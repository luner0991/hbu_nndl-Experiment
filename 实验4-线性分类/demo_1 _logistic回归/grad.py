import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLR(nn.Module):
    def __init__(self, input_dim):
        super(ModelLR, self).__init__()
        # 存放线性层参数
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['w'] = nn.Parameter(torch.zeros(input_dim, 1))
        # 如果需要使用不同的初始化方法，请取消下面这行的注释
        # self.params['w'] = nn.Parameter(torch.normal(0, 0.01, (input_dim, 1)))
        # 将线性层的偏置参数初始化为0
        self.params['b'] = nn.Parameter(torch.zeros(1))
        # 存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None

    def forward(self, inputs):
        self.X = inputs
        # 线性计算
        score = torch.matmul(inputs, self.params['w']) + self.params['b']
        # Logistic 函数
        self.outputs = torch.sigmoid(score)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]
        """
        N = labels.shape[0]
        # 计算偏导数
        self.grads['w'] = -1 / N * torch.matmul(self.X.t(), (labels - self.outputs))
        self.grads['b'] = -1 / N * torch.sum(labels - self.outputs)