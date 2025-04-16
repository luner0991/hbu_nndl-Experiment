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
            weight = torch.tensor([[0., 1.], [2., 3.]], dtype=torch.float32)
        else:
            weight = torch.tensor(weight, dtype=torch.float32)
        # 创建卷积核参数
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
        # 在输入矩阵周围填充零
        if self.padding > 0:
            X = torch.nn.functional.pad(X, (self.padding, self.padding, self.padding, self.padding))

        u, v = self.weight.shape  # 获取卷积核形状
        B, M, N = X.shape  # 获取数据的形状

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
                region = X[:, row_start:row_start + u, col_start:col_start + v]
                # 计算卷积操作
                output[:, i, j] = torch.sum(region * self.weight, dim=(1, 2))

        return output

# 测试代码
torch.manual_seed(100)
inputs = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]])
conv2d = Conv2D(kernel_size=2, stride=2, padding=1)
outputs = conv2d(inputs)
print("Input: \n", inputs)
print("Output: \n", outputs)
