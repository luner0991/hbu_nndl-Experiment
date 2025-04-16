import torch
import torch.nn as nn
import torch.nn.functional as F
class ModelSR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelSR, self).__init__()
        self.params = nn.ParameterDict({
            'W': nn.Parameter(torch.zeros(input_dim, output_dim)),
            'b': nn.Parameter(torch.zeros(output_dim))
        })

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        输入：
            - inputs: shape=[N, D], N是样本数量，D是特征维度
        输出：
            - outputs：预测值，shape=[N, C]，C是类别数
        """
        # 线性计算
        score = torch.matmul(inputs, self.params['W']) + self.params['b']
        # Softmax 函数
        outputs = F.softmax(score, dim=1)
        return outputs

# 随机生成1条长度为4的数据
inputs = torch.randn(1, 4)
print('Input is:', inputs)
# 实例化模型，这里令输入长度为4，输出类别数为3
model = ModelSR(input_dim=4, output_dim=3)
outputs = model(inputs)
print('Output is:', outputs)