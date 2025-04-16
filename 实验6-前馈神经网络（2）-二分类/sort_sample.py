import torch
from data import make_moons
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 定义两层神经网络
class TwoLayerNN(nn.Module):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 输入层到隐藏层
        self.fc2 = nn.Linear(16, 1)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.sigmoid(self.fc2(x))  # 输出层使用Sigmoid激活函数
        return x

# 训练模型并保存损失值
def train_model(X, y, num_epochs=1000, learning_rate=0.01):
    model = TwoLayerNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X).view(-1, 1)  # 确保输出形状为[N, 1]
        loss = criterion(outputs, y)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model, losses

# 计算准确率
def calculate_accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X).view(-1)  # [N, 1] 转换为 [N]
        predicted = (outputs >= 0.5).float()  # 根据阈值预测类别
        accuracy = (predicted == y.view(-1)).float().mean().item()  # 计算准确率
    return accuracy

# 可视化分类边界
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid).squeeze().numpy()

    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(10, 5))
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.5)
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=y.tolist(), marker='*')
    plt.xlim(-3, 4)
    plt.ylim(-3, 4)
    plt.title("分类边界")
    plt.show()

# 可视化损失图像
def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失图像')
    plt.legend()
    plt.grid()
    plt.show()

# 主函数
if __name__ == "__main__":
    # 数据集构建
    n_samples = 1000
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.2)

    # 划分数据集
    num_train = 640  # 训练集样本数量
    num_dev = 160  # 验证集样本数量
    num_test = 200  # 测试集样本数量

    # 根据指定数量划分数据集
    X_train, y_train = X[:num_train], y[:num_train]  # 训练集
    X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]  # 验证集
    X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]  # 测试集

    # 调整标签的形状，将其转换为[N, 1]的格式
    y_train = y_train.reshape([-1, 1])
    y_dev = y_dev.reshape([-1, 1])
    y_test = y_test.reshape([-1, 1])

    # 可视化生成的数据集
    plt.figure(figsize=(5, 5))  # 设置图形大小
    plt.scatter(x=X[:, 0], y=X[:, 1], marker='*', c=y, cmap='viridis')  # 绘制散点图
    plt.xlim(-3, 4)  # 设置x轴范围
    plt.ylim(-3, 4)  # 设置y轴范围
    plt.grid(True, linestyle='--', alpha=0.3)  # 添加网格
    plt.show()  # 显示图形

    # 训练模型
    model, losses = train_model(torch.FloatTensor(X_train), torch.FloatTensor(y_train))

    # 可视化分类边界
    plot_decision_boundary(model, torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    plot_loss(losses)

    # 计算并打印准确率
    train_accuracy = calculate_accuracy(model, torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    dev_accuracy = calculate_accuracy(model, torch.FloatTensor(X_dev), torch.FloatTensor(y_dev))
    test_accuracy = calculate_accuracy(model, torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    print(f'训练集准确率: {train_accuracy:.4f}')
    print(f'验证集准确率: {dev_accuracy:.4f}')
    print(f'测试集准确率: {test_accuracy:.4f}')
