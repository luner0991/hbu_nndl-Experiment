import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLR(nn.Module):
    def __init__(self, input_dim):
        super(ModelLR, self).__init__()
        # 定义模型参数并初始化
        self.params = {}
        # 初始化权重参数为0，形状为 [input_dim, 1]
        # 可选：使用正态分布初始化权重
        self.params['w'] = nn.Parameter(torch.normal(0, 0.01, (input_dim, 1)))
        # 初始化偏置参数为0，形状为 [1]
        self.params['b'] = nn.Parameter(torch.zeros(1))

    def __call__(self, inputs):
        """
        定义 __call__ 方法以便直接调用模型，等同于调用 forward 方法
        输入:
            - inputs: 输入数据
        输出:
            - 模型的预测结果
        """
        return self.forward(inputs)

    def forward(self, inputs):
        """
        前向传播函数
        输入:
            - inputs: shape=[N, D]，N 为样本数量，D 为特征维度
        输出:
            - outputs: 预测标签为1的概率，shape=[N, 1]
        """
        # 线性计算，使用初始化的权重和偏置
        score = torch.matmul(inputs, self.params['w']) + self.params['b']
        # 使用 sigmoid 函数将线性输出转化为概率
        outputs = torch.sigmoid(score)
        return outputs

# 固定随机种子，保持每次运行结果一致
torch.manual_seed(0)
# 随机生成3条长度为4的数据
inputs = torch.randn(size=[3,4])
print('Input is:', inputs)
# 实例化模型
model = ModelLR(4)
outputs = model(inputs)
print('Output is:', outputs)
