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
test_num = 10
X_train, y_train = create_data(sin, interval, train_num,noise=0)
X_test, y_test = create_data(sin, interval, test_num, noise=0)  # 无噪声测试集
X_underlying = torch.linspace(interval[0], interval[1], steps=100).reshape(-1, 1)  # 真实值
y_underlying = sin(X_underlying.numpy())

# 训练误差和测试误差
training_errors = []
test_errors = []

    # 遍历多项式阶数
    for i in range(9):
        model = LinearModel(i)

        X_train_transformed = polynomial_basis_function(X_train, i)
        X_test_transformed = polynomial_basis_function(X_test, i)
        X_underlying_transformed = polynomial_basis_function(X_underlying, i)

        model = optimizer_lsm(model, X_train_transformed, y_train)  # 拟合得到参数

        y_train_pred = model(X_train_transformed).squeeze()
        y_test_pred = model(X_test_transformed).squeeze()
        y_underlying_pred = model(X_underlying_transformed).squeeze()

        train_mse = mse_loss(y_train_pred, y_train).item()
        training_errors.append(train_mse)

        test_mse = mse_loss(y_test_pred, y_test).item()
        test_errors.append(test_mse)

    # 输出误差
    print("训练误差: \n", training_errors)
    print("测试误差: \n", test_errors)

    # 绘制图片
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.plot(training_errors, '-.', mfc="none", mec='#e4007f', ms=10, c='#e4007f', label="训练误差")
    plt.plot(test_errors, '--', mfc="none", mec='#f19ec2', ms=10, c='#f19ec2', label="测试误差")
    plt.legend(fontsize='x-large')
    plt.xlabel("多项式阶数")
    plt.ylabel("均方误差 (MSE)")
    plt.title("训练误差与测试误差")
    plt.show()
