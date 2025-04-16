import torch
import torch.nn as nn

class ModelSR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelSR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# 输入维度
input_dim = 4
# 类别数
output_dim = 3
# 实例化模型
model = ModelSR(input_dim=input_dim, output_dim=output_dim)
# 假设有一个输入张量 x
x = torch.randn(1, input_dim)  # 示例输入数据
# 使用模型进行前向传播
output = model(x)
# 输出是模型对输入的预测
print(output)
