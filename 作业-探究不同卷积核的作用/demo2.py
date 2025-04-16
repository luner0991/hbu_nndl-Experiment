import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取彩色图像并转换为灰度图
file_path = 'rabbit.jpeg'
im = Image.open(file_path).convert('L')  # 读入一张灰度图的图片
im = np.array(im, dtype='float32')  # 转换为矩阵

# 可视化原图
plt.imshow(im.astype('uint8'), cmap='gray')
plt.title('原图')
plt.axis('off')
plt.show()

# 转换为四维张量
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))

# 定义卷积层
conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)

# 自定义锐化卷积核
sharpen_kernel = np.array([[-2, -1, 0],
                            [-1, 1, 1],
                            [0, 1, 2]], dtype='float32')
# 改为更强的锐化效果
# sharpen_kernel = np.array([[-1, -1, -1],
#                             [-1, 9, -1],
#                             [-1, -1, -1]], dtype='float32')

sharpen_kernel = sharpen_kernel.reshape((1, 1, 3, 3))
conv.weight.data = torch.from_numpy(sharpen_kernel)
sharp_image = conv(im).data.squeeze().numpy()

# 可视化锐化结果
plt.imshow(sharp_image, cmap='gray', vmin=0, vmax=255)
plt.title('锐化图')
plt.axis('off')
plt.show()

# 自定义模糊卷积核
blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype='float32') / 5  # 平均模糊
# 改为更强的模糊效果
# blur_kernel = np.array([[1, 1, 1],
#                         [1, -6, 1],
#                         [1, 1, 1]], dtype='float32')

blur_kernel = blur_kernel.reshape((1, 1, 3, 3))
conv.weight.data = torch.from_numpy(blur_kernel)
blurred_image = conv(Variable(im)).data.squeeze().numpy()

# 可视化模糊结果
plt.imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
plt.title('模糊图')
plt.axis('off')
plt.show()

# 自定义边缘检测卷积核
edge_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype='float32')
# 改为更强的边缘检测效果
# edge_kernel = np.array([[-1, 0, 1],
#                          [-2, 0, 2],
#                          [-1, 0, 1]], dtype='float32')

edge_kernel = edge_kernel.reshape((1, 1, 3, 3))
conv.weight.data = torch.from_numpy(edge_kernel)
edge_image = conv(Variable(im)).data.squeeze().numpy()

# 可视化边缘检测结果
plt.imshow(edge_image, cmap='gray', vmin=0, vmax=255)
plt.title('边缘检测图')
plt.axis('off')
plt.show()
