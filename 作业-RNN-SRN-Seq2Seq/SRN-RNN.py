'''
@Function: 使用nn.RNN实现SRN
@Author: lxy
@Date: 2024/11/24
'''
import torch
# 设置批处理大小
batch_size = 1
# 设置序列长度
seq_len = 3
# 输入序列的维度
input_size = 2
# 隐藏层的维度
hidden_size = 2
# 输出层的维度
output_size = 2
# RNN层的数量
num_layers = 1

# 创建RNN实例
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# 初始化参数
for name, param in cell.named_parameters():
    if name.startswith('weight'):
        torch.nn.init.ones_(param)
    else:
        torch.nn.init.zeros_(param)

# 线性层 将隐藏状态映射到输出
linear = torch.nn.Linear(hidden_size, output_size)
# 初始化线性层的权重为1
linear.weight.data = torch.Tensor([[1,1],[1,1]])
# 初始化线性层的偏置为0
linear.bias.data = torch.Tensor([0.0])

# 创建输入序列
'''
三维，形状(3, 1, 2)
第一维：序列长度（seq_len），这里是3
第二维：批次大小（batch_size），这里是1
第三维：输入特征的维度（input_size），这里是2
'''
inputs = torch.Tensor([[[1,1]],[[1,1]],[[2,2]]])
# 初始化隐藏状态为0，这里需要考虑RNN层的数量
hidden = torch.zeros(num_layers, batch_size, hidden_size)
# 通过RNN处理输入序列，并更新隐藏状态
out, hidden = cell(inputs, hidden)

# 打印第一个时间步的输入、隐藏状态和输出
print("第1时刻：")
print(f'Input : {inputs[0]}')
print(f'hidden: {[0 , 0]}')
print(f'Output: {linear(out[0])}')
print('======================================')
# 打印第二个时间步的输入、隐藏状态和输出
print("第2时刻：")
print(f'Input : {inputs[1]}')
print(f'hidden: {out[0]}')
print(f'Output: {linear(out[1])}')
print('=======================================')
# 打印第三个时间步的输入、隐藏状态和输出
print("第3时刻：")
print(f'Input : {inputs[2]}')
print(f'hidden: {out[1]}')
print(f'Output: {linear(out[2])}')