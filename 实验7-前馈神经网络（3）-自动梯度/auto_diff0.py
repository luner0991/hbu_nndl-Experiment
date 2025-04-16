from data import make_moons
from nndl import accuracy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from Runner2_2 import RunnerV2_2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Model_MLP_L2_V2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model_MLP_L2_V2, self).__init__()
        # 定义第一层线性层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 使用正态分布初始化权重和偏置
        self.fc1.weight.data = torch.normal(mean=0.0, std=1.0, size=self.fc1.weight.data.size())
        self.fc1.bias.data.fill_(0.0)  # 常量初始化偏置为0
        # 定义第二层线性层
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2.weight.data = torch.normal(mean=0.0, std=1.0, size=self.fc2.weight.data.size())
        self.fc2.bias.data.fill_(0.0)  # 常量初始化偏置为0
        # 定义Logistic激活函数
        self.act_fn = torch.sigmoid
        self.layers = [self.fc1, self.act_fn, self.fc2,self.act_fn]


    # 前向计算
    def forward(self, inputs):
        z1 = self.fc1(inputs)
        a1 = self.act_fn(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn(z2)
        return a2
# 数据集构建
n_samples = 1000
X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.2)
# 划分数据集
num_train = 640  # 训练集样本数量
num_dev = 160    # 验证集样本数量
num_test = 200   # 测试集样本数量
# 根据指定数量划分数据集
X_train, y_train = X[:num_train], y[:num_train]  # 训练集
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]  # 验证集
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]  # 测试集
# 调整标签的形状，将其转换为[N, 1]的格式
y_train = y_train.reshape([-1, 1])
y_dev = y_dev.reshape([-1, 1])
y_test = y_test.reshape([-1, 1])
 #n 可视化生成的数据集
plt.figure(figsize=(5, 5))  # 设置图形大小
plt.scatter(x=X[:, 0], y=X[:, 1], marker='*', c=y, cmap='viridis')  # 绘制散点图
plt.xlim(-3, 4)  # 设置x轴范围
plt.ylim(-3, 4)  # 设置y轴范围
plt.grid(True, linestyle='--', alpha=0.3)  # 添加网格
#plt.show()  # 显示图形
# 定义训练参数
epoch_num = 1000  # 训练轮数
model_saved_dir = "best_model.pdparams"  # 模型保存目录
# 网络参数
input_size = 2  # 输入层维度为2
hidden_size = 6 # 隐藏层维度为5
output_size = 1  # 输出层维度为1
# 定义多层感知机模型
model = Model_MLP_L2_V2(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# 定义损失函数
loss_fn =F.binary_cross_entropy
# 定义优化器，设置学习率
learning_rate = 9
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)
# 定义评价方法
metric = accuracy
# 实例化RunnerV2_1类，并传入训练配置
runner = RunnerV2_2(model, optimizer, metric, loss_fn)
# 训练模型
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=epoch_num, log_epochs=50, save_dir=model_saved_dir)
# 加载训练好的模型----测试
runner.load_model("best_model.pdparams")

# 在测试集上对模型进行评价
score, loss = runner.evaluate([X_test, y_test])  # 评估模型性能vv吃饭的
# 打印测试集的准确率和损失
print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))
