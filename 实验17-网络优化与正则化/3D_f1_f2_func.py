'''
@Function:两个函数的3D可视化
@Author；lxy
@Date:2024/12/19
'''
import numpy as np
from matplotlib import pyplot as plt
import torch
from op import Op

#1.函数3D可视化
#第一个函数
class OptimizedFunction3D1(Op):
    def __init__(self):
        super(OptimizedFunction3D1, self).__init__()
        self.params = {'x': 0}
        self.grads = {'x': 0}

    def forward(self, x):
        self.params['x'] = x
        return x[0] ** 2 + x[1] ** 2 + x[1] ** 3 + x[0] * x[1]

    def backward(self):
        x = self.params['x']
        gradient1 = 2 * x[0] + x[1]
        gradient2 = 2 * x[1] + 3 * x[1] ** 2 + x[0]
        grad1 = torch.Tensor([gradient1])
        grad2 = torch.Tensor([gradient2])
        self.grads['x'] = torch.cat([grad1, grad2])

#第二个函数
class OptimizedFunction3D2(Op):
    def __init__(self):
        super(OptimizedFunction3D2, self).__init__()
        self.params = {'x': 0}
        self.grads = {'x': 0}

    def forward(self, x):
        self.params['x'] = x
        return x[0] * x[0] / 20 + x[1] * x[1] / 1  # x[0] ** 2 + x[1] ** 2 + x[1] ** 3 + x[0] * x[1]

    def backward(self):
        x = self.params['x']
        gradient1 = 2 * x[0] / 20
        gradient2 = 2 * x[1] / 1
        grad1 = torch.Tensor([gradient1])
        grad2 = torch.Tensor([gradient2])
        self.grads['x'] = torch.cat([grad1, grad2])


# 使用numpy.meshgrid生成x1,x2矩阵，矩阵的每一行为[-3, 3]，以0.1为间隔的数值
x1 = np.arange(-3, 3, 0.1)
x2 = np.arange(-3, 3, 0.1)
x1, x2 = np.meshgrid(x1, x2)#网格
init_x = torch.Tensor(np.array([x1, x2]))

model1 = OptimizedFunction3D1()#实例化
model2 = OptimizedFunction3D2()#实例化

fig = plt.figure()
#绘制第一个函数的3D图像
ax1 = fig.add_subplot(121, projection='3d')
y1 = model1.forward(init_x)
ax1.plot_surface(x1, x2, y1.detach(), cmap='rainbow')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('z1')
ax1.set_title('Optimized Function1 3D')

#绘制第二个函数的3D图像
ax2 = fig.add_subplot(122, projection='3d')
y2 = model2.forward(init_x)
ax2.plot_surface(x1, x2, y2.detach(), cmap='coolwarm')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('z2')
ax2.set_title('Optimized Function2 3D')

plt.show()
