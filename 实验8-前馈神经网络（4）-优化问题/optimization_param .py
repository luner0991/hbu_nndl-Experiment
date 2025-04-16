'''
@author: lxy
@function: The Impact of Zero Weight Initialization
@date: 2024/10/31
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, normal_, uniform_
import torch
from data import make_moons
from nndl import accuracy
from Runner2_2 import RunnerV2_2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Model_MLP_L2_V4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model_MLP_L2_V4, self).__init__()
        # 定义第一个线性层，输入特征数为 input_size，输出特征数为 hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        '''
         weight为权重参数属性,bias为偏置参数属性，这里使用'torch.nn.init.constant_'进行常量初始化
        '''
        # 初始化第一个线性层的权重和偏置为 0
        constant_(self.fc1.weight, 0.0)
        constant_(self.fc1.bias, 0.0)
        # 定义第二个线性层，输入特征数为 hidden_size，输出特征数为 output_size
        self.fc2 = nn.Linear(hidden_size, output_size)
        # 初始化第二个线性层的权重和偏置为 0
        constant_(self.fc2.weight, 0.0)
        constant_(self.fc2.bias, 0.0)

        self.act_fn = F.sigmoid

    # 前向计算
    def forward(self, inputs):
        z1 = self.fc1(inputs)
        a1 = self.act_fn(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn(z2)
        return a2

def print_weight(runner):
    print('The weights of the Layers：')
    # 通过 enumerate() 可以同时获取参数的索引 i 和参数的内容 item
    for i, item in enumerate(runner.model.named_parameters()):
        print(item)
        print('=========================')

# =============================数据集=======================
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
# ================================训练模型===========================
input_size = 2
hidden_size = 5
output_size = 1
model = Model_MLP_L2_V4(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# 设置损失函数
loss_fn = F.binary_cross_entropy

# 设置优化器
learning_rate = 0.2
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# 设置评价指标
metric = accuracy

# 其他参数
epoch = 2000
saved_path = 'best_model.pdparams'

# 实例化RunnerV2_2类，并传入训练配置
runner = RunnerV2_2(model, optimizer, metric, loss_fn)

runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=5, log_epochs=50, save_path="best_model.pdparams",
             custom_print_log=print_weight)

# ===========可视化函数===============
def plot(runner, fig_name):
    plt.figure(figsize=(10, 5))
    epochs = [i for i in range(0, len(runner.train_scores))]
    plt.subplot(1, 2, 1)
    plt.plot(epochs, runner.train_loss, color='#e4007f', label="Train loss")
    plt.plot(epochs, runner.dev_loss, color='#f19ec2', linestyle='--', label="Dev loss")
    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, runner.train_scores, color='#e4007f', label="Train accuracy")
    plt.plot(epochs, runner.dev_scores, color='#f19ec2', linestyle='--', label="Dev accuracy")
    # 绘制坐标轴和图例
    plt.ylabel("score", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig(fig_name)
    plt.show()
plot(runner, 'fw-acc.pdf')