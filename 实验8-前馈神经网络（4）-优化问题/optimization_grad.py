'''
@author: lxy
@function: Exploration and Optimization of the Gradient Vanishing Problem
@date: 2024/10/31
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, normal_
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data import make_moons
from nndl import accuracy
from Runner2_2 import RunnerV2_2

class Model_MLP_L5(nn.Module):
    def __init__(self, input_size, output_size, act='sigmoid', w_init=nn.init.normal_, b_init=nn.init.constant_):
        super(Model_MLP_L5, self).__init__()
        self.fc1 = nn.Linear(input_size, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 3)
        self.fc5 = nn.Linear(3, output_size)

        # 定义激活函数
        if act == 'sigmoid':
            self.act = F.sigmoid
        elif act == 'relu':
            self.act = F.relu
        elif act == 'lrelu':
            self.act = F.leaky_relu
        else:
            raise ValueError("Please enter sigmoid, relu or lrelu!")

        # 初始化权重和偏置
        self.init_weights(w_init, b_init)

    # 初始化线性层权重和偏置参数
    def init_weights(self, w_init, b_init):
        for m in self.children():
            if isinstance(m, nn.Linear):
                w_init(m.weight, mean=0.0, std=0.01)  # 对权重进行初始化
                b_init(m.bias, 1.0)  # 对偏置进行初始化

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.act(outputs)
        outputs = self.fc2(outputs)
        outputs = self.act(outputs)
        outputs = self.fc3(outputs)
        outputs = self.act(outputs)
        outputs = self.fc4(outputs)
        outputs = self.act(outputs)
        outputs = self.fc5(outputs)
        outputs = F.sigmoid(outputs)
        return outputs


def print_grads(runner, grad_norms):
    """ 打印模型每一层的梯度并计算其L2范数。 """
    print("The gradient of the Layers:")
    for name, param in runner.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()  # 计算L2范数
            grad_norms[name].append(grad_norm)  # 记录L2范数
            print(f'Layer: {name}, Gradient Norm: {grad_norm}')



# 可视化梯度L2范数
def plot_grad_norms(grad_norms_sigmoid, grad_norms_relu):
    layers = list(grad_norms_sigmoid.keys())
    sigmoid_norms = [np.mean(grad_norms_sigmoid[layer]) for layer in layers]
    relu_norms = [np.mean(grad_norms_relu[layer]) for layer in layers]

    x = np.arange(len(layers))

    plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid_norms, marker='o', label='Sigmoid', color='b')
    plt.plot(x, relu_norms, marker='o', label='ReLU', color='r')

    plt.ylabel('Gradient L2 Norm')
    plt.title('Gradient L2 Norm by different Activation Function')
    plt.xticks(x, layers)
    plt.legend()
    # 设置 y 轴为对数坐标
    plt.yscale('log')
    # 设置 y 轴的范围
    plt.ylim(1e-8, 1)  # 设置下限为 1e-8，上限为 1
    # 设置 y 轴的刻度
    plt.yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,1e-9,1e-10,1e-11])

    plt.grid()
    plt.tight_layout()
    plt.show()


# =============================数据集=======================
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
# =====================使用sigmoid激活函数训练=====================
torch.manual_seed(111)
lr = 0.01
model = Model_MLP_L5(input_size=2, output_size=1, act='sigmoid')
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
loss_fn = F.binary_cross_entropy
metric = accuracy

# 初始化L2范数记录字典
grad_norms_sigmoid = {name: [] for name, _ in model.named_parameters()}

# 实例化Runner类
runner = RunnerV2_2(model, optimizer, metric, loss_fn)
print("使用sigmoid函数为激活函数时：")
runner.train([X_train, y_train], [X_dev, y_dev],
             num_epochs=1, log_epochs=None,
             save_path="best_model.pdparams",
             custom_print_log=lambda runner: print_grads(runner, grad_norms_sigmoid))

# =====================使用ReLU激活函数训练=====================
torch.manual_seed(102)
model = Model_MLP_L5(input_size=2, output_size=1, act='relu')
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
loss_fn = F.binary_cross_entropy

# 初始化L2范数记录字典
grad_norms_relu = {name: [] for name, _ in model.named_parameters()}

# 实例化Runner类
runner = RunnerV2_2(model, optimizer, metric, loss_fn)
print("使用ReLU函数为激活函数时：")
runner.train([X_train, y_train], [X_dev, y_dev],
             num_epochs=1, log_epochs=None,
             save_path="best_model.pdparams",
             custom_print_log=lambda runner: print_grads(runner, grad_norms_relu))

# 绘制梯度范数
plot_grad_norms(grad_norms_sigmoid, grad_norms_relu)
