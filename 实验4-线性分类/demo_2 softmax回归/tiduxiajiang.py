import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelSR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelSR, self).__init__()
        # 将线性层的权重参数全部初始化为0
        self.params = {
            'W': nn.Parameter(torch.zeros(size=[input_dim, output_dim])),
            'b': nn.Parameter(torch.zeros(size=[output_dim]))
        }
        # 存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def forward(self, inputs):
        self.X = inputs
        # 线性计算
        score = torch.matmul(self.X, self.params['W']) + self.params['b']
        # Softmax 函数
        self.outputs = F.softmax(score, dim=1)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]，其中N为样本数量
        """
        # 计算偏导数
        N = labels.shape[0]
        labels = labels.view(-1).long()  # 确保标签为一维
        one_hot_labels = F.one_hot(labels, num_classes=self.output_dim).float()  # 独热编码

        # 计算梯度
        self.grads['W'] = -1 / N * torch.matmul(self.X.t(), (one_hot_labels - self.outputs))
        self.grads['b'] = -1 / N * torch.sum(one_hot_labels - self.outputs, dim=0)

# 测试一下
if __name__ == "__main__":
    input_dim = 4  # 输入特征维度
    output_dim = 3  # 输出类别数量
    model = ModelSR(input_dim, output_dim)

    # 随机生成输入数据和标签
    inputs = torch.randn(5, input_dim)  # 5个样本
    labels = torch.tensor([0, 1, 2, 0, 1]).view(-1, 1)  # 标签

    # 前向传播
    outputs = model(inputs)
    print("Outputs:", outputs)

    # 反向传播
    model.backward(labels)
    print("Gradients W:", model.grads['W'])
    print("Gradients b:", model.grads['b'])
