import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据集构建
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 将数据集划分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.36, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.555, random_state=42)

# 转换为 Tensor
X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train).unsqueeze(1)
X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val).unsqueeze(1)
X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test).unsqueeze(1)

# 构建 DataLoader
train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_data = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_data = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


# 2. 模型构建
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = torch.randn(input_dim, output_dim, requires_grad=True)  # 权重矩阵
        self.b = torch.randn(output_dim, requires_grad=True)  # 偏置向量
        self.input = None

    def forward(self, x):
        self.input = x
        return torch.matmul(x, self.W) + self.b

    def backward(self, grad_output):
        grad_input = torch.matmul(grad_output, self.W.T)
        self.W.grad = torch.matmul(self.input.T, grad_output)
        self.b.grad = grad_output.sum(0)
        return grad_input


class Logistic:
    def forward(self, x):
        self.out = 1 / (1 + torch.exp(-x))  # Sigmoid 激活
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)  # Logistic 函数的导数


# 3. 损失函数
class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -torch.mean(y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8))

    def backward(self):
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) + 1e-8)


# 4. 优化器
class Optimizer:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W.data -= self.lr * layer.W.grad
            if hasattr(layer, 'b'):
                layer.b.data -= self.lr * layer.b.grad


# 5. Runner 类
class RunnerV2_1:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, x, y):
        y_pred = self.model.forward(x)
        loss = self.loss_fn.forward(y_pred, y)
        grad_loss = self.loss_fn.backward()
        self.model.backward(grad_loss)
        self.optimizer.step()
        return loss

    def train(self, train_loader, val_loader, epochs=2000):
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')


# 6. 两层神经网络模型
class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.logistic1 = Logistic()
        self.linear2 = LinearLayer(hidden_dim, output_dim)
        self.logistic2 = Logistic()

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.logistic1.forward(x)
        x = self.linear2.forward(x)
        return self.logistic2.forward(x)

    def backward(self, grad_loss):
        grad_loss = self.logistic2.backward(grad_loss)
        grad_loss = self.linear2.backward(grad_loss)
        grad_loss = self.logistic1.backward(grad_loss)
        self.linear1.backward(grad_loss)


# 7. 模型训练
model = TwoLayerNet(input_dim=2, hidden_dim=10, output_dim=1)
loss_fn = BinaryCrossEntropy()
optimizer = Optimizer([model.linear1, model.linear2], lr=0.01)

runner = RunnerV2_1(model, loss_fn, optimizer)
runner.train(train_data, val_data, epochs=500)


# 8. 测试模型
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model.forward(x_batch)
            predicted = (y_pred > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')


evaluate(model, test_data)

# 数据集可视化函数
def plot_decision_boundary(model, X, y, title="Decision Boundary", show_boundary=True):
    # 创建一个二维网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # 用模型预测网格点上的分类
    grid_points = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model.forward(grid_points)
    Z = Z.reshape(xx.shape)

    # 可选地画分类边界
    if show_boundary:
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="RdYlBu", alpha=0.7)

    # 画散点图
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.show()

# 绘制训练集上的分类边界（不绘制分类边界）
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().reshape(-1)
plot_decision_boundary(model, X_train_np, y_train_np, show_boundary=False)
# 绘制训练集上的分类边界
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().reshape(-1)
plot_decision_boundary(model, X_train_np, y_train_np)
