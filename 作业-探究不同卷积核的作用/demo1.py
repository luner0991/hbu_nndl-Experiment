# -*- coding: gbk -*-
'''
@author: lxy
@function: Explore the effects different  kernels
@date: 2024/10/20
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 创建一个包含三幅图像的图形窗口
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 第一张图：7x6的图像，右边三列为255
x1 = torch.zeros((7, 6))    # 创建7x6的全零图像
x1[:, 3:6] = 255            # 将右边三列设置为255
axs[0].imshow(x1, cmap='Greys_r')
axs[0].set_title('First Image')

# 第二张图：14x12的图像，左下和右上区域为255
x2 = torch.zeros((14, 12))    # 创建14x12的全零图像
x2[7:14, 0:6] = 255           # 左下区域设置为255
x2[0:7, 6:12] = 255           # 右上区域设置为255
axs[1].imshow(x2, cmap='Greys_r')
axs[1].set_title('Second Image')

# 第三张图：9x9的图像，对角线和反对角线为255
x3 = torch.zeros((9, 9))     # 创建9x9的全零图像
for i in range(1, 8):         # 设置黑色像素点
    x3[i, i] = 255
    x3[i, 8 - i] = 255

axs[2].imshow(x3, cmap='Greys_r')
axs[2].set_title('Third Image')

# 显示所有图像
plt.tight_layout()
plt.show()
'''
卷积操作：
'''
# 定义卷积核 (-1, 1)
kernel1 = torch.tensor([[-1, 1]], dtype=torch.float32).reshape(1, 1, 1, 2)  # 创建卷积核，并调整形状为(1, 1, 1, 2)
# 定义卷积核为[[ -1 ], [ 1 ]]
kernel2 = torch.tensor([[ -1 ], [ 1 ]], dtype=torch.float32).reshape(1, 1, 2, 1)
# 定义卷积核为[[1,-1],[-1,1]]
kernel3 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).reshape(1, 1, 2, 2)
# 将图像转换为适合卷积的形状
x1_reshaped = x1.unsqueeze(0).unsqueeze(0)  # 添加两个维度，使形状变为(1, 1, 7, 6)，适配卷积层输入格式
x2_reshaped = x2.unsqueeze(0).unsqueeze(0)  # 将 x2 转换为 (1, 1, 14, 12)
x3_reshaped = x3.unsqueeze(0).unsqueeze(0)  # 将 x3 转换为 (1, 1, 9, 9)

'''
对图1的卷积操作--使用kernel1
'''
conv1_x1 = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # 定义一个卷积层，输入和输出通道数均为1，卷积核大小为(1, 2)
conv1_x1.weight.data = kernel1  # 将自定义的卷积核赋值给卷积层的权重
x1_conv = conv1_x1(x1_reshaped)  # 使用卷积层对转换后的x1进行卷积操作，得到卷积结果
plt.figure(figsize=(5, 5))
plt.imshow(x1_conv.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x1 Convolution Result --- kernel1[[-1, 1]]')
plt.show()
'''
对图1的卷积操作--使用kernel2
'''
conv2_x1 = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)  # 卷积核大小设置为 (2, 1)
conv2_x1.weight.data = kernel2  # 将转置的卷积核赋值给卷积层
x1_conv_kernel2 = conv2_x1(x1_reshaped)  # 对 x1 进行卷积操作
plt.figure(figsize=(5, 5))
plt.imshow(x1_conv_kernel2.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x1 Convolution Result --- kernel2[[ -1 ], [ 1 ]]')
plt.show()
'''
对图2的卷积操作--使用kernel1
'''
conv1_x2 = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # 定义卷积层，使用kernel1
conv1_x2.weight.data = kernel1  # 将卷积核赋值给卷积层
x2_conv_kernel1 = conv1_x2(x2_reshaped)  # 对 x2 进行卷积操作
plt.figure(figsize=(5, 5))
plt.imshow(x2_conv_kernel1.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x2 Convolution Result --- kernel1[[-1, 1]]')
plt.show()
'''
对图2的卷积操作--使用kernel2
'''
conv2_x2 = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)  # 定义卷积层，使用kernel2
conv2_x2.weight.data = kernel2  # 将卷积核赋值给卷积层
x2_conv_kernel2 = conv2_x2(x2_reshaped)  # 对 x2 进行卷积操作
plt.figure(figsize=(5, 5))
plt.imshow(x2_conv_kernel2.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x2 Convolution Result --- kernel2[[ -1 ], [ 1 ]]')
plt.show()
'''
 对 图3的卷积操作 ---使用kernel1
'''
conv1_x3 = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
conv1_x3.weight.data = kernel1
x3_conv1 = conv1_x3(x3_reshaped)  # 进行卷积操作
plt.figure(figsize=(5, 5))
plt.imshow(x3_conv1.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x3 Convolution Result --- kernel1[[1, -1]]')
plt.show()
'''
 对 图3的卷积操作 ---使用kernel2
'''
conv2_x3 = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)
conv2_x3.weight.data = kernel2
x3_conv2 = conv2_x3(x3_reshaped)  # 进行卷积操作
plt.figure(figsize=(5, 5))
plt.imshow(x3_conv2.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x3 Convolution Result --- kernel2[[1], [-1]]')
plt.show()
'''
 对 图3的卷积操作 ---使用kernel3
'''
conv3_x3 = nn.Conv2d(1, 1, kernel_size=(2, 2), bias=False)
conv3_x3.weight.data = kernel3
x3_conv3 = conv3_x3(x3_reshaped)  # 进行卷积操作
plt.figure(figsize=(5, 5))
plt.imshow(x3_conv3.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x3 Convolution Result --- kernel3[[1, -1], [-1, 1]]')
plt.show()