'''
@Function: 实现序列到序列：给出hello -->预测ohlol
@Author: lxy
@Date: 2024/11/24
'''
import torch
import torch.nn as nn
import torch.optim as optim

# 定义参数
input_size = 4 # 输入特征的维度：对应的是 one-hot 编码的大小（4个字符）
hidden_size = 4 # 隐藏层维度
batch_size = 1  # 批次大小（这里是1，表示一次只处理一个字符）

# 字符到索引的映射
idx2char = ['e', 'h', 'l', 'o']

# 输入和标签数据
x_data = [1, 0, 2, 2, 3] # 对应字符 ['h', 'e', 'l', 'l', 'o']
y_data = [3, 1, 2, 3, 2] # 对应目标字符 ['o', 'h', 'l', 'o', 'l']

# 将字符映射到one-hot编码
one_hot_lookup = [
    [1, 0, 0, 0],  # 'e' 对应 [1, 0, 0, 0]
    [0, 1, 0, 0],  # 'h' 对应 [0, 1, 0, 0]
    [0, 0, 1, 0],  # 'l' 对应 [0, 0, 1, 0]
    [0, 0, 0, 1]   # 'o' 对应 [0, 0, 0, 1]
]
# 将输入数据 x_data 转换为 one-hot 编码形式
x_one_hot = [one_hot_lookup[x] for x in x_data]

# 将输入和标签转换为PyTorch张量
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  # (序列长度, 批次大小, 输入维度)
labels = torch.LongTensor(y_data).view(-1, 1)  # (序列长度, 1)，每个标签对应一个字符的索引

# 定义RNN模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        # RNNCell 处理每个输入，并返回新的隐藏状态
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

# 初始化模型、损失函数和优化器
net = Model(input_size, hidden_size, batch_size)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.1)  # 使用 Adam 优化器

# 训练模型
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden() # 初始化隐藏状态
    print('Predicted string: ', end='')
    # 遍历输入序列和标签
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)  # 将输入传入网络并获取新的隐藏状态
        loss += criterion(hidden, label)  # 计算损失并累加
        _, idx = hidden.max(dim=1)   # 获取预测的字符索引（最大值）
        print(idx2char[idx.item()], end='')  # 输出对应的字符
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))