import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# sin函数
def sin(x):
    return np.sin(2 * np.pi * x)

# 创建数据集
def create_data(func, interval, num, noise=0.5):
    x = np.random.rand(num, 1) * (interval[1] - interval[0]) + interval[0]
    y = func(x)
    epsilon = np.random.normal(0, noise, y.shape)
    y = y + epsilon
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 多项式基函数
def polynomial_basis_function(X, degree):
    return torch.cat([X**d for d in range(degree + 1)], dim=1)

# 线性模型类
class LinearModel(nn.Module):
    def __init__(self, degree):
        super(LinearModel, self).__init__()
        self.params = nn.Parameter(torch.randn(degree + 1))  # 多项式系数

    def forward(self, X):
        return torch.matmul(X, self.params)

# 损失函数
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# 优化函数
def optimizer_lsm(model, X, y):
    # 最小二乘法优化
    w = torch.matmul(torch.inverse(torch.matmul(X.T, X)), torch.matmul(X.T, y))
    model.params.data = w
    return model

# 创建训练集和测试集
interval = (0, 1)
train_num = 15
X_train, y_train = create_data(sin, interval, train_num)
X_underlying = torch.linspace(interval[0], interval[1], steps=100).reshape(-1, 1)  # 修正这里
y_underlying = sin(X_underlying.numpy())

# 设置多项式阶数
degrees = [0, 1, 3, 8]

# 绘图
plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees):
    model = LinearModel(degree)
    X_train_transformed = polynomial_basis_function(X_train, degree)
    X_underlying_transformed = polynomial_basis_function(X_underlying, degree)

    model = optimizer_lsm(model, X_train_transformed, y_train)  # 拟合得到参数

    y_underlying_pred = model(X_underlying_transformed).detach().numpy().squeeze()

    # 计算损失
    loss = mse_loss(torch.tensor(y_underlying_pred), torch.tensor(y_underlying.flatten()))
    print(f"多项式阶数 {degree} 的损失: {loss.item()}")

    # 绘制图像
    plt.subplot(2, 2, i + 1)
    plt.scatter(X_train.numpy(), y_train.numpy(), facecolor="none", edgecolor='#e4007f', s=50, label="训练数据")
    plt.plot(X_underlying.numpy(), y_underlying, c='#000000', label=r"$\sin(2\pi x)$")
    plt.plot(X_underlying.numpy(), y_underlying_pred, c='#f19ec2', label="拟合曲线")
    plt.ylim(-2, 1.5)
    plt.annotate("M={}".format(degree), xy=(0.95, -1.4))

plt.legend(loc='lower left', fontsize='x-large')
plt.suptitle("不同多项式阶的拟合结果")
plt.tight_layout()
plt.show()
