'''
@Author: lxy
@Function: Implement Conv2D Operator with Stride and Padding
@Date: 2024/11/06
'''
import torch
import torch.nn as nn
class Conv2D(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, weight=None):
        super(Conv2D, self).__init__()
        # 初始化卷积核
        if weight is None:
            weight = torch.tensor([[0., 1., 2.], [3., 4. ,5.],[6.,7.,8.]], dtype=torch.float32)
        else:
            weight = torch.tensor(weight, dtype=torch.float32)
        # 创建卷积核参数,权重被初始化为Parameter，表明在训练过程中进行优化
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.stride = stride  # 步长
        self.padding = padding  # 填充

    def forward(self, X):
        """
        输入：
            - X：输入矩阵，shape=[B, M, N]，B为样本数量 ，M表示矩阵高度，N表示矩阵宽度
        输出：
            - output：输出矩阵 形状为 [B, output_height, output_width]
        """
        # 零填充
        new_X = torch.zeros([X.shape[0], X.shape[1] + 2 * self.padding, X.shape[2] + 2 * self.padding])
        # 将原始输入X放置在new_X的中央区域，在new_X的左侧和顶部添加了填充
        new_X[:, self.padding:X.shape[1] + self.padding, self.padding:X.shape[2] + self.padding] = X
        u, v = self.weight.shape  # 获取卷积核形状
        B, M, N = new_X.shape  # 获取填充后数据的形状
        # 计算输出矩阵的高度和宽度
        output_height = (M - u) // self.stride + 1
        output_width = (N - v) // self.stride + 1
        output = torch.zeros((B, output_height, output_width), dtype=X.dtype)  # 初始化输出矩阵
        # 遍历输出矩阵的每个位置，计算卷积值
        for i in range(output_height):
            for j in range(output_width):
                # 通过步长控制子矩阵的位置
                row_start = i * self.stride
                col_start = j * self.stride
                # 提取 X 中当前卷积核位置覆盖的子矩阵
                region = new_X[:, row_start:row_start + u, col_start:col_start + v]
                # 计算卷积操作A
                output[:, i, j] = torch.sum(region * self.weight, dim=(1, 2))
        return output


# 测试代码
torch.manual_seed(100)
inputs = torch.randn(size=[2, 8, 8])  # 随机生成特征图 2通道，形状为8*8
print(f'inputs为:{inputs}')
conv2d_padding = Conv2D(kernel_size=3,stride=1, padding=1)  # 将填充设置为1 进行卷积
outputs = conv2d_padding(inputs)
print(f"When kernel_size=3, padding=1 stride=1, input's shape: {inputs.shape}, output's shape: {outputs.shape}")
conv2d_stride = Conv2D(kernel_size=3, stride=2, padding=1)
outputs = conv2d_stride(inputs)
print(f"When kernel_size=3, padding=1 stride=2, input's shape: {inputs.shape}, output's shape: {outputs.shape}")

