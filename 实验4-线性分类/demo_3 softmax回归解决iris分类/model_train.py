import torch
import torch.nn as nn
from Runner2 import RunnerV2
import torch.nn.functional as F

class ModelSR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelSR, self).__init__()
        # 将线性层的权重参数全部初始化为0
        self.params = {
            'W': nn.Parameter(torch.zeros(size=[input_dim, output_dim])),
            'b': nn.Parameter(torch.zeros(size=[output_dim]))
        }
        # 存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def forward(self, inputs):
        self.X = inputs
        # 线性计算
        score = torch.matmul(self.X, self.params['W']) + self.params['b']
        # Softmax 函数
        self.outputs = F.softmax(score, dim=1)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]，其中N为样本数量
        """
        # 计算偏导数
        N = labels.shape[0]
        labels = labels.view(-1).long()  # 确保标签为一维并转换为长整型

        one_hot_labels = F.one_hot(labels, num_classes=self.output_dim).float()  # 独热编码

        # 计算梯度
        self.grads['W'] = -1 / N * torch.matmul(self.X.t(), (one_hot_labels - self.outputs))
        self.grads['b'] = -1 / N * torch.sum(one_hot_labels - self.outputs, dim=0)


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, C]，N为样本数量，C为类别数量
            - labels：真实标签，shape=[N]
        输出：
            - 损失值：shape=[1]
        """
        # 将标签转换为长整型
        labels = labels.view(-1).long()
        N = predicts.shape[0]  # 样本数量
        loss = 0.0

        # 计算损失
        for i in range(N):
            index = labels[i]  # 获取当前样本的标签
            loss -= torch.log(predicts[i][index])  # 计算交叉熵损失

        return loss / N  # 返回平均损失
from abc import abstractmethod
# 优化器基类
class Optimizer(object):
    def __init__(self, init_lr, model):
        """
        优化器类初始化
        """
        # 初始化学习率，用于参数更新的计算
        self.init_lr = init_lr
        # 指定优化器需要优化的模型
        self.model = model

    @abstractmethod
    def step(self):
        """
        定义每次迭代如何更新参数
        """
        pass
class SimpleBatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(SimpleBatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        # 参数更新
        # 遍历所有参数，按照公式(3.8)和(3.9)更新参数
        if isinstance(self.model.params, dict):
            for key in self.model.params.keys():
                self.model.params[key] = self.model.params[key] - self.init_lr * self.model.grads[key]
def accuracy(preds, labels):
    """
    输入：
        - preds：预测值，二分类时，shape=[N, 1]，N为样本数量，多分类时，shape=[N, C]，C为类别数量
        - labels：真实标签，shape=[N, 1]
    输出：
        - 准确率：shape=[1]
    """
    # 判断是二分类任务还是多分类任务，preds.shape[1]=1时为二分类任务，preds.shape[1]>1时为多分类任务
    if preds.shape[1] == 1:
        # 二分类时，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0
        # 使用 'torch.round' 进行四舍五入，将概率值转换为二进制标签
        preds = torch.round(preds)
    else:
        # 多分类时，使用 'torch.argmax' 计算最大元素索引作为类别
        preds = torch.argmax(preds, dim=1)

    # 计算准确率
    correct = (preds == labels).sum().item()
    #     print("correct:",correct)
    accuracy = correct / len(labels)
    #     print("shape of labels:",labels.shape)
    #     print("labels:",labels)
    #     print("shape of preds:",preds.shape)
    #     print("preds:",preds)
    return accuracy

#===============数据集===============
from sklearn.datasets import load_iris
import pandas
import numpy as np

iris_features = np.array(load_iris().data, dtype=np.float32)
iris_labels = np.array(load_iris().target, dtype=np.int32)
print(pandas.isna(iris_features).sum())
print(pandas.isna(iris_labels).sum())
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt #可视化工具

# 箱线图查看异常值分布
def boxplot(features):
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # 连续画几个图片
    plt.figure(figsize=(5, 5), dpi=200)
    # 子图调整
    plt.subplots_adjust(wspace=0.6)
    # 每个特征画一个箱线图
    for i in range(4):
        plt.subplot(2, 2, i+1)
        # 画箱线图
        plt.boxplot(features[:, i],
                    showmeans=True,
                    whiskerprops={"color":"#E20079", "linewidth":0.4, 'linestyle':"--"},
                    flierprops={"markersize":0.4},
                    meanprops={"markersize":1})
        # 图名
        plt.title(feature_names[i], fontdict={"size":5}, pad=2)
        # y方向刻度
        plt.yticks(fontsize=4, rotation=90)
        plt.tick_params(pad=0.5)
        # x方向刻度
        plt.xticks([])
    #plt.savefig('ml-vis.pdf')
    plt.show()

boxplot(iris_features)

#===============划分数据集==================
import copy
import torch

# 加载数据集
def load_data(shuffle=True):
    """
    加载鸢尾花数据
    输入：
        - shuffle：是否打乱数据，数据类型为bool
    输出：
        - X：特征数据，shape=[150,4]
        - y：标签数据, shape=[150]
    """
    # 加载原始数据
    X = np.array(load_iris().data, dtype=np.float32)
    y = np.array(load_iris().target, dtype=np.int32)

    X = torch.tensor(X)
    y = torch.tensor(y)

    # 数据归一化
    X_min,_ = torch.min(X, axis=0)
    X_max,_ = torch.max(X, axis=0)
    X = (X-X_min) / (X_max-X_min)

    # 如果shuffle为True，随机打乱数据
    if shuffle:
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]
    return X, y
# 固定随机种子
torch.manual_seed(102)

num_train = 120
num_dev = 15
num_test = 15

X, y = load_data(shuffle=True)
print("X shape: ", X.shape, "y shape: ", y.shape)
X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]
# 启动训练
# 学习率
lr = 0.2
# 输入维度
input_dim = 4
# 类别数
output_dim = 3
# 实例化模型
model = ModelSR(input_dim=input_dim, output_dim=output_dim)
# 梯度下降法
optimizer = SimpleBatchGD(init_lr=lr, model=model)
# 交叉熵损失
loss_fn = MultiCrossEntropyLoss()
# 准确率
metric = accuracy

# 实例化RunnerV2
runner = RunnerV2(model, optimizer, metric, loss_fn)

#启动训练
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=100, log_epochs=1000, save_path="best_model.pdparams")

# 可视化观察训练集与验证集的指标变化情况
def plot(runner,fig_name):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    epochs = [i for i in range(len(runner.train_scores))]
    # 绘制训练损失变化曲线
    plt.plot(epochs, runner.train_loss, color='#e4007f', label="Train loss")
    # 绘制评价损失变化曲线
    plt.plot(epochs, runner.dev_loss, color='#f19ec2', linestyle='--', label="Dev loss")
    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.subplot(1,2,2)
    # 绘制训练准确率变化曲线
    plt.plot(epochs, runner.train_scores, color='#e4007f', label="Train accuracy")
    # 绘制评价准确率变化曲线
    plt.plot(epochs, runner.dev_scores, color='#f19ec2', linestyle='--', label="Dev accuracy")
    # 绘制坐标轴和图例
    plt.ylabel("score", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
# 可视化观察训练集与验证集的准确率变化情况
plot(runner,fig_name='linear-acc2.pdf')

#=================m模型评价=======================
# 加载最优模型
runner.load_model('best_model.pdparams')
# 模型评价
score, loss = runner.evaluate([X_test, y_test])
print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))