'''
@Author: lxy
@Function: 分别用自定义卷积算子和torch.nn.Conv2d()编程实现特定卷积运算
@Date: 2024/11/07
'''
import torch
import torch.nn as nn

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,weight =None,bias= None):
        super(Conv2D, self).__init__()
        # 初始化卷积核
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        # 权重维度为 [out_channels, in_channels, kernel_height, kernel_width]
        if weight is None:
            weight = torch.randn((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32)
        else:
            weight = torch.tensor(weight, dtype=torch.float32)
        if bias is None:
            bias = torch.zeros(out_channels,1)
        else:
            bias = torch.tensor(bias, dtype=torch.float32)
        # 创建卷积核参数
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.bias = nn.Parameter(bias,requires_grad=True)

    # 单个通道卷积操作
    def single_forward(self, X, weight):
        # 零填充输入
        new_X = torch.zeros((X.shape[0], X.shape[1] + 2 * self.padding, X.shape[2] + 2 * self.padding))
        new_X[:, self.padding:X.shape[1] + self.padding, self.padding:X.shape[2] + self.padding] = X

        u, v = weight.shape  # 卷积核形状
        output_h = (new_X.shape[1] - u) // self.stride + 1
        output_w = (new_X.shape[2] - v) // self.stride + 1
        output = torch.zeros((X.shape[0], output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                output[:, i, j] = torch.sum(
                    new_X[:, i * self.stride:i * self.stride + u, j * self.stride:j * self.stride + v] * weight,
                    dim=(1, 2)
                )
        return output

    def forward(self, X):
        """
        输入：
            - X：输入张量，shape=[B, C_in, H, W]，
              其中 B 是批大小，C_in 是输入通道数，H 是高度，W 是宽度
        输出：
            - output：输出张量，shape=[B, C_out, output_height, output_width]
        """
        feature_maps = []
        for w, b in zip(self.weight, self.bias):  # 遍历每个输出通道
            multi_outs = []
            for i in range(self.in_channels):  # 对每个输入通道计算卷积
                single = self.single_forward(inputs[:, i, :, :], w[i])
                multi_outs.append(single)
            # 将各通道卷积结果相加并添加偏置
            feature_map = torch.sum(torch.stack(multi_outs), dim=0) + b
            feature_maps.append(feature_map)
            # 将所有输出通道的结果堆叠
        out = torch.stack(feature_maps, dim=1)
        return out

# 测试代码
torch.manual_seed(100)
# 输入特征图
inputs = torch.tensor([[[[0, 1, 1, 0, 2],
                         [2, 2, 2, 2, 1],
                         [1, 0, 0, 2, 0],
                         [0, 1, 1, 0, 0],
                         [1, 2, 0, 0, 2]],

                        [[1, 0, 2, 2, 0],
                         [0, 0, 0, 2, 0],
                         [1, 2, 1, 2, 1],
                         [1, 0, 0, 0, 0],
                         [1, 2, 1, 1, 1]],

                        [[2, 1, 2, 0, 0],
                         [1, 0, 0, 1, 0],
                         [0, 2, 1, 0, 1],
                         [0, 1, 2, 2, 2],
                         [2, 1, 0, 0, 1]]]], dtype=torch.float32)
# 自定义卷积核权重
weight = torch.tensor([[[[-1, 1, 0],  # 第一组
                         [0, 1, 0],
                         [0, 1, 1]],

                        [[-1, -1, 0],
                         [0, 0, 0],
                         [0, -1, 0]],

                        [[0, 0, -1],
                         [0, 1, 0],
                         [1, -1, -1]]],

                       [[[1, 1, -1],  # 第二组
                         [-1, -1, 1],
                         [0, -1, 1]],

                        [[0, 1, 0],
                         [-1, 0, -1],
                         [-1, 1, 0]],

                        [[-1, 0, 0],
                         [-1, 0, 1],
                         [-1, 0, 0]]]], dtype=torch.float32)

# 自定义卷积核偏置
bias = torch.tensor([1., 0.])
# 创建一个卷积层，出入通道3 输出通道2 卷积核大小3*3
conv2d = Conv2D(in_channels=3, out_channels=2, kernel_size=3,stride=2, padding=1)
# 为卷积层设置偏置项，第一组卷积核的bias=1,第二组卷积核的bias=0
conv2d.bias = nn.Parameter(bias)
# 为卷积层设置权重(为卷积核矩阵赋值)
conv2d.weight = nn.Parameter(weight)
'''
输出通道数 = 卷积核个数 out_channels
输出高度 = (输入高度 - 卷积核高度) / 步长 + 1 = (3 - 2) / 1 + 1 = 2
输出宽度 = (输入宽度 - 卷积核宽度) / 步长 + 1 = (3 - 2) / 1 + 1 = 2
'''
print("inputs shape:",inputs.shape)
outputs = conv2d(inputs)
print("Conv2D outputs shape:",outputs.shape)

# 比较与pytorch API运算结果
conv2d_pytorch = nn.Conv2d(in_channels=2, out_channels=3, kernel_size = 2,stride=2, padding=1)
# 手动修改权重
with torch.no_grad():  # 在不计算梯度的情况下修改权重
    conv2d_pytorch.weight = nn.Parameter(weight)  # 重新定义权重
    # 手动设置偏置
    conv2d_pytorch.bias = nn.Parameter(bias)  # 重新定义偏置
outputs_pytorch = conv2d_pytorch(inputs)

# 自定义算子运算结果
print('Conv2D outputs:', outputs)
# pytorch API运算结果
print('nn.Conv2D outputs:', outputs_pytorch)