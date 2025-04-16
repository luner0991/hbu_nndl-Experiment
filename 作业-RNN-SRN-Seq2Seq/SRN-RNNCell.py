'''
@Function: 使用nn.RNNCell实现SRN
@Author: lxy
@Date: 2024/11/24
'''
import torch
batch_size = 1
seq_len = 3  # 序列长度（多少时间步）
input_size = 2  # 输入序列维度
hidden_size = 2  # 隐藏层维度
output_size = 2  # 输出层维度

# # 创建RNNCell实例
cell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)
# 初始化参数 https://zhuanlan.zhihu.com/p/342012463
'''
RNN的weight和bias封装在parameters中，且需要对weight和bias分开初始化
'''
for name,param in cell.named_parameters():
    if name.startswith('weight'): # 初始化weight
        torch.nn.init.ones_(param)
    else: # 初始化bias
        torch.nn.init.zeros_(param)

# 线性层-> 将隐藏状态映射到输出
linear = torch.nn.Linear(hidden_size,output_size)
# 初始化线性层的权重为1
linear.weight.data = torch.Tensor([[1,1],[1,1]])
# 初始化线性层的偏置为0
linear.bias.data = torch.Tensor([0.0])

# 定义输入序列
'''
三维，形状(3, 1, 2)
第一维：序列长度（seq_len），这里是3
第二维：批次大小（batch_size），这里是1
第三维：输入特征的维度（input_size），这里是2
'''
seq = torch.Tensor([[[1,1]],[[1,1]],[[2,2]]])
# 初始化隐藏状态和输出为0
hidden = torch.zeros(batch_size,hidden_size)
output = torch.zeros(batch_size,output_size)
# 遍历序列中的每一个时间步
for idx,input in enumerate(seq):
    print('===========================')
    print(f"第{idx+1}时刻：")
    print(f'Input :{input}')
    print(f'hidden :{hidden}')
    hidden = cell(input,hidden) # 使用RNNCell处理当前输入，并更新隐藏状态
    output = linear(hidden) # # 使用线性层将隐藏状态转换为输出
    print(f"output :{output}")