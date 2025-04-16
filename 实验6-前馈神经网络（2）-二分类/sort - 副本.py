from data import make_moons
from nndl import Op
from nndl import accuracy
from nndl import Optimizer
import numpy as np
import torch
from Runner2_1 import RunnerV2_1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 定义共同的父类 Op

class Linear(Op):
    """
    线性层（全连接层）。

    参数：
        - input_size: 输入特征的数量
        - output_size: 输出特征的数量
        - name: 层的名称
        - weight_init: 权重初始化函数
        - bias_init: 偏置初始化函数

    属性：
        - params: 存储权重和偏置
        - grads: 存储梯度
        - inputs: 前向传播的输入
    """
    def __init__(self, input_size, output_size, name, weight_init=torch.randn, bias_init=torch.zeros):
        self.params = {}
        self.params['W'] = weight_init(size=(input_size, output_size))
        self.params['b'] = bias_init(size=(1, output_size))
        self.inputs = None
        self.grads = {}
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        outputs = torch.matmul(self.inputs, self.params['W']) + self.params['b']  # 线性变换
        return outputs

    def backward(self, grads):
        self.grads['W'] = torch.matmul(self.inputs.T, grads)  # 计算权重梯度
        self.grads['b'] = torch.sum(grads, axis=0)  # 计算偏置梯度
        return torch.matmul(grads, self.params['W'].T)  # 返回上层梯度


'''
# ReLU 激活函数
class ReLU(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = {}
        self.name = "ReLU"

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = torch.maximum(inputs, torch.tensor(0.0))
        return self.outputs

    def backward(self, grads):
        return grads * (self.inputs > 0).float()
'''
class Logistic(Op):
    """
    Sigmoid 激活函数。

    属性：
        - inputs: 前向传播的输入
        - outputs: 前向传播的输出
        - params: 存储模型参数
        - name: 层的名称
    """
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = {}
        self.name = "Logistic"

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + torch.exp(-inputs))  # Sigmoid 函数
        self.outputs = outputs
        return self.outputs

    def backward(self, grads):
        outputs_grad_inputs = self.outputs * (1.0 - self.outputs)  # Sigmoid 导数
        return grads * outputs_grad_inputs  # 返回梯度


class BinaryCrossEntropyLoss(Op):
    """
    二分类交叉熵损失函数。

    属性：
        - predicts: 模型预测值
        - labels: 真实标签
        - num: 样本数量
        - model: 需要计算梯度的模型
    """
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None
        self.model = model

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = -1. / self.num * (torch.matmul(self.labels.t(), torch.log(self.predicts)) +
                                 torch.matmul((1 - self.labels.t()), torch.log(1 - self.predicts)))
        loss = torch.squeeze(loss, axis=1)  # 压缩维度
        return loss

    def backward(self):
        loss_grad_predicts = -1.0 * (self.labels / self.predicts -
                                     (1 - self.labels) / (1 - self.predicts)) / self.num  # 计算梯度
        self.model.backward(loss_grad_predicts)  # 反向传播


class BatchGD(Optimizer):
    """
    批量梯度下降优化器。

    参数：
        - init_lr: 初始学习率
        - model: 需要优化的模型
    """
    def __init__(self, init_lr, model):
        super(BatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        # 更新参数
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]  # 更新权重和偏置


class Model_MLP_L2(Op):
    """
    二层全连接神经网络模型。

    参数：
        - input_size: 输入层特征数量
        - hidden_size: 隐藏层特征数量
        - output_size: 输出层特征数量

    属性：
        - layers: 存储模型的所有层
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size, name="fc1")
        self.act_fn1 = Logistic()  # 激活函数
        self.fc2 = Linear(hidden_size, output_size, name="fc2")
        self.act_fn2 = Logistic()  # 激活函数
        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2]  # 按顺序存储层

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        z1 = self.fc1(X)  # 第一层前向传播
        a1 = self.act_fn1(z1)  # 第一层激活
        z2 = self.fc2(a1)  # 第二层前向传播
        a2 = self.act_fn2(z2)  # 第二层激活
        return a2  # 返回输出

    def backward(self, loss_grad_a2):
        # 反向传播
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)  # 第二层反向传播
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)  # 第一层反向传播
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)  # 激活函数反向传播
        self.fc1.backward(loss_grad_z1)  # 更新权重和偏置


# 实例化模型
model = Model_MLP_L2(input_size=5, hidden_size=10, output_size=1)
# 随机生成1条长度为5的数据
X = torch.rand(size=(1, 5))
result = model(X)
print("result: ", result)

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

# 可视化生成的数据集
plt.figure(figsize=(5, 5))  # 设置图形大小
plt.scatter(x=X[:, 0], y=X[:, 1], marker='*', c=y, cmap='viridis')  # 绘制散点图
plt.xlim(-3, 4)  # 设置x轴范围
plt.ylim(-3, 4)  # 设置y轴范围
plt.grid(True, linestyle='--', alpha=0.3)  # 添加网格
plt.show()  # 显示图形

# 设置随机种子以确保结果可重复
torch.manual_seed(111)

# 定义训练参数
epoch_num = 1000  # 训练轮数
model_saved_dir = "model"  # 模型保存目录

# 网络参数
input_size = 2  # 输入层维度为2
hidden_size = 5  # 隐藏层维度为5
output_size = 1  # 输出层维度为1

# 定义多层感知机模型
model = Model_MLP_L2(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# 定义损失函数
loss_fn = BinaryCrossEntropyLoss(model)

# 定义优化器，设置学习率
learning_rate = 1
optimizer = BatchGD(learning_rate, model)

# 定义评价方法
metric = accuracy

# 实例化RunnerV2_1类，并传入训练配置
runner = RunnerV2_1(model, optimizer, metric, loss_fn)

# 训练模型
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=epoch_num, log_epochs=50, save_dir=model_saved_dir)

# 打印训练集和验证集的损失
plt.figure()  # 创建新的图形
plt.plot(range(epoch_num), runner.train_loss, color="#8E004D", label="Train loss")  # 绘制训练损失
plt.plot(range(epoch_num), runner.dev_loss, color="#E20079", linestyle='--', label="Dev loss")  # 绘制验证损失
plt.xlabel("epoch", fontsize='x-large')  # 设置x轴标签
plt.ylabel("loss", fontsize='x-large')  # 设置y轴标签
plt.legend(fontsize='large')  # 显示图例
plt.show()  # 显示损失图

# 加载训练好的模型
runner.load_model(model_saved_dir)

# 在测试集上对模型进行评价
score, loss = runner.evaluate([X_test, y_test])  # 评估模型性能

# 打印测试集的准确率和损失
print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))

import math

# 均匀生成40000个数据点
x1, x2 = torch.meshgrid(torch.linspace(-math.pi, math.pi, 200), torch.linspace(-math.pi, math.pi, 200))
x = torch.stack([torch.flatten(x1), torch.flatten(x2)], axis=1)  # 将生成的点堆叠成二维数组

# 使用模型进行预测
y = runner.predict(x)  # 预测类别
y = torch.squeeze((y >= 0.5).to(torch.float32), axis=-1)  # 将概率值转化为类别标签

# 绘制类别区域
plt.ylabel('x2')  # 设置y轴标签
plt.xlabel('x1')  # 设置x轴标签
plt.scatter(x[:, 0].tolist(), x[:, 1].tolist(), c=y.tolist(), cmap=plt.cm.Spectral)  # 绘制类别区域

# 可视化训练集、验证集和测试集数据
plt.scatter(X_dev[:, 0].tolist(), X_dev[:, 1].tolist(), marker='*', c=torch.squeeze(y_dev, axis=-1).tolist())  # 绘制验证集
plt.scatter(X_test[:, 0].tolist(), X_test[:, 1].tolist(), marker='*', c=torch.squeeze(y_test, axis=-1).tolist())  # 绘制测试集
plt.show()  # 显示最终图形


