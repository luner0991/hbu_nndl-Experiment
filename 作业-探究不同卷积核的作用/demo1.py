# -*- coding: gbk -*-
'''
@author: lxy
@function: Explore the effects different  kernels
@date: 2024/10/20
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ʹ��TkAgg���
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ����һ����������ͼ���ͼ�δ���
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# ��һ��ͼ��7x6��ͼ���ұ�����Ϊ255
x1 = torch.zeros((7, 6))    # ����7x6��ȫ��ͼ��
x1[:, 3:6] = 255            # ���ұ���������Ϊ255
axs[0].imshow(x1, cmap='Greys_r')
axs[0].set_title('First Image')

# �ڶ���ͼ��14x12��ͼ�����º���������Ϊ255
x2 = torch.zeros((14, 12))    # ����14x12��ȫ��ͼ��
x2[7:14, 0:6] = 255           # ������������Ϊ255
x2[0:7, 6:12] = 255           # ������������Ϊ255
axs[1].imshow(x2, cmap='Greys_r')
axs[1].set_title('Second Image')

# ������ͼ��9x9��ͼ�񣬶Խ��ߺͷ��Խ���Ϊ255
x3 = torch.zeros((9, 9))     # ����9x9��ȫ��ͼ��
for i in range(1, 8):         # ���ú�ɫ���ص�
    x3[i, i] = 255
    x3[i, 8 - i] = 255

axs[2].imshow(x3, cmap='Greys_r')
axs[2].set_title('Third Image')

# ��ʾ����ͼ��
plt.tight_layout()
plt.show()
'''
���������
'''
# �������� (-1, 1)
kernel1 = torch.tensor([[-1, 1]], dtype=torch.float32).reshape(1, 1, 1, 2)  # ��������ˣ���������״Ϊ(1, 1, 1, 2)
# ��������Ϊ[[ -1 ], [ 1 ]]
kernel2 = torch.tensor([[ -1 ], [ 1 ]], dtype=torch.float32).reshape(1, 1, 2, 1)
# ��������Ϊ[[1,-1],[-1,1]]
kernel3 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).reshape(1, 1, 2, 2)
# ��ͼ��ת��Ϊ�ʺϾ������״
x1_reshaped = x1.unsqueeze(0).unsqueeze(0)  # �������ά�ȣ�ʹ��״��Ϊ(1, 1, 7, 6)���������������ʽ
x2_reshaped = x2.unsqueeze(0).unsqueeze(0)  # �� x2 ת��Ϊ (1, 1, 14, 12)
x3_reshaped = x3.unsqueeze(0).unsqueeze(0)  # �� x3 ת��Ϊ (1, 1, 9, 9)

'''
��ͼ1�ľ������--ʹ��kernel1
'''
conv1_x1 = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # ����һ������㣬��������ͨ������Ϊ1������˴�СΪ(1, 2)
conv1_x1.weight.data = kernel1  # ���Զ���ľ���˸�ֵ��������Ȩ��
x1_conv = conv1_x1(x1_reshaped)  # ʹ�þ�����ת�����x1���о���������õ�������
plt.figure(figsize=(5, 5))
plt.imshow(x1_conv.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x1 Convolution Result --- kernel1[[-1, 1]]')
plt.show()
'''
��ͼ1�ľ������--ʹ��kernel2
'''
conv2_x1 = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)  # ����˴�С����Ϊ (2, 1)
conv2_x1.weight.data = kernel2  # ��ת�õľ���˸�ֵ�������
x1_conv_kernel2 = conv2_x1(x1_reshaped)  # �� x1 ���о������
plt.figure(figsize=(5, 5))
plt.imshow(x1_conv_kernel2.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x1 Convolution Result --- kernel2[[ -1 ], [ 1 ]]')
plt.show()
'''
��ͼ2�ľ������--ʹ��kernel1
'''
conv1_x2 = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # �������㣬ʹ��kernel1
conv1_x2.weight.data = kernel1  # ������˸�ֵ�������
x2_conv_kernel1 = conv1_x2(x2_reshaped)  # �� x2 ���о������
plt.figure(figsize=(5, 5))
plt.imshow(x2_conv_kernel1.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x2 Convolution Result --- kernel1[[-1, 1]]')
plt.show()
'''
��ͼ2�ľ������--ʹ��kernel2
'''
conv2_x2 = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)  # �������㣬ʹ��kernel2
conv2_x2.weight.data = kernel2  # ������˸�ֵ�������
x2_conv_kernel2 = conv2_x2(x2_reshaped)  # �� x2 ���о������
plt.figure(figsize=(5, 5))
plt.imshow(x2_conv_kernel2.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x2 Convolution Result --- kernel2[[ -1 ], [ 1 ]]')
plt.show()
'''
 �� ͼ3�ľ������ ---ʹ��kernel1
'''
conv1_x3 = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
conv1_x3.weight.data = kernel1
x3_conv1 = conv1_x3(x3_reshaped)  # ���о������
plt.figure(figsize=(5, 5))
plt.imshow(x3_conv1.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x3 Convolution Result --- kernel1[[1, -1]]')
plt.show()
'''
 �� ͼ3�ľ������ ---ʹ��kernel2
'''
conv2_x3 = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)
conv2_x3.weight.data = kernel2
x3_conv2 = conv2_x3(x3_reshaped)  # ���о������
plt.figure(figsize=(5, 5))
plt.imshow(x3_conv2.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x3 Convolution Result --- kernel2[[1], [-1]]')
plt.show()
'''
 �� ͼ3�ľ������ ---ʹ��kernel3
'''
conv3_x3 = nn.Conv2d(1, 1, kernel_size=(2, 2), bias=False)
conv3_x3.weight.data = kernel3
x3_conv3 = conv3_x3(x3_reshaped)  # ���о������
plt.figure(figsize=(5, 5))
plt.imshow(x3_conv3.squeeze().detach().numpy(), cmap='Greys_r')
plt.title('x3 Convolution Result --- kernel3[[1, -1], [-1, 1]]')
plt.show()