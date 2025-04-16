import torch
import torch.nn as nn

# 创建一个Conv2d层实例
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
# 准备输入数据
input_data = torch.randn(1, 1, 2, 2)  # 假设输入是一个2x2的单通道图像
# 应用卷积层
output_data = conv_layer(input_data)
# 查看输出数据的形状
print(output_data.shape)
# 访问权重
weights = conv_layer.weight
# 访问偏置
biases = conv_layer.bias
print(f"权重是{weights}\n偏置是{biases}")

# 创建一个MaxPool2d层
maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
# 假设输入数据是卷积层的输出
input_data = output_data
# 应用最大池化层
output_data_maxpool = maxpool_layer(input_data)
print(f"池化前{output_data}\n最大池化后{output_data_maxpool}")
# 创建一个AvgPool2d层
avgpool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
# 假设输入数据是卷积层的输出
input_data = output_data
# 应用平均池化层
output_data_avgpool = avgpool_layer(input_data)
print(f"池化前{output_data}\n平均池化后{output_data_avgpool}")
