import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from runner import Runner

# 定义 train_test_split 函数
def train_test_split(X, y, train_percent=0.8):
    """将数据集分割为训练集和测试集"""
    n = len(X)  # 数据集大小
    shuffled_indices = np.random.permutation(n)  # 随机排列索引
    train_set_size = int(n * train_percent)  # 训练集大小

    train_indices = shuffled_indices[:train_set_size]  # 训练集索引
    test_indices = shuffled_indices[train_set_size:]  # 测试集索引

    X_train = X.values[train_indices]  # 训练集特征
    y_train = y.values[train_indices]  # 训练集目标
    X_test = X.values[test_indices]  # 测试集特征
    y_test = y.values[test_indices]  # 测试集目标

    return X_train, X_test, y_train, y_test

# 读取数据
data = pd.read_csv('boston_house_prices.csv')

# 分割特征和目标变量
X = data.drop(['MEDV'], axis=1)  # 特征
y = data['MEDV']  # 目标变量

# 调用函数划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ==========================特征工程--归一化处理================================
X_train_min = X_train.min(axis=0)
X_train_max = X_train.max(axis=0)
X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 使目标变量为列向量
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 训练集构造
train_dataset = (X_train_tensor, y_train_tensor)
# 测试集构造
test_dataset = (X_test_tensor, y_test_tensor)

# ==================模型构建======================
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层

    def forward(self, x):
        return self.linear(x)

# 超参数
input_size = X_train_tensor.shape[1]
model = LinearRegressionModel(input_size)

# 损失函数和评估指标
mse_loss = nn.MSELoss()
# ==================Runner类实例化===============================
runner = Runner(model, loss_fn=mse_loss, metric=mse_loss)
# 模型保存文件夹
saved_dir = 'models'
# 启动训练
runner.train(train_dataset, model_dir=saved_dir)
# 打印出训练得到的权重和偏置
weights = model.linear.weight.detach().numpy().flatten()
b = model.linear.bias.detach().numpy().item()
# 获取特征名
feature_names = X.columns.tolist()
for i in range(len(weights)):
    print(f'特征 {feature_names[i]}: 权重: {weights[i]}')
print(f'偏置: {b}')

# 测试模型
test_loss = runner.evaluate(test_dataset)
print(f'测试损失: {test_loss.item():.4f}')
# 模型预测
runner.load_model(saved_dir)
pred = runner.predict(X_test[:1])
print("真实房价：",y_test[:1].item())
print("预测的房价：",pred.item())

