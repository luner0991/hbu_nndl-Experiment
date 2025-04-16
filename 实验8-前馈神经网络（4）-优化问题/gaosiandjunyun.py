import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

# 使用'torch.normal'实现高斯分布采样，其中'mean'为高斯分布的均值，'std'为高斯分布的标准差，'shape'为输出形状
gausian_weights = torch.normal(mean=0.0, std=1.0, size=[10000])
# 使用'torch.uniform'实现在[min,max)范围内的均匀分布采样，其中'shape'为输出形状
uniform_weights = torch.Tensor(10000)
uniform_weights.uniform_(-1,1)
gausian_weights=gausian_weights.numpy()
uniform_weights=uniform_weights.numpy()
print(uniform_weights)
# 绘制两种参数分布
plt.figure()
plt.subplot(1,2,1)
plt.title('Gausian Distribution')
plt.hist(gausian_weights, bins=200, density=True, color='#f19ec2')
plt.subplot(1,2,2)
plt.title('Uniform Distribution')
plt.hist(uniform_weights, bins=200, density=True, color='#e4007f')
plt.savefig('fw-gausian-uniform.pdf')
plt.show()