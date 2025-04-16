import torch
import torch.nn as nn
import torch.nn.functional as F
from Runner2 import RunnerV2  # 导入 RunnerV2 类
class ModelLR(nn.Module):
    def __init__(self, input_dim):
        super(ModelLR, self).__init__()
        # 存放线性层参数
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['w'] = nn.Parameter(torch.zeros(input_dim, 1))
        # 如果需要使用不同的初始化方法，请取消下面这行的注释
        # self.params['w'] = nn.Parameter(torch.normal(0, 0.01, (input_dim, 1)))
        # 将线性层的偏置参数初始化为0
        self.params['b'] = nn.Parameter(torch.zeros(1))
        # 存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None

    def forward(self, inputs):
        self.X = inputs
        # 线性计算
        score = torch.matmul(inputs, self.params['w']) + self.params['b']
        # Logistic 函数
        self.outputs = torch.sigmoid(score)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]
        """
        N = labels.shape[0]
        # 计算偏导数
        self.grads['w'] = -1 / N * torch.matmul(self.X.t(), (labels - self.outputs))
        self.grads['b'] = -1 / N * torch.sum(labels - self.outputs)
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.predicts = None
        self.labels = None
        self.num = None

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, 1]，N为样本数量
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]

        # 计算二元交叉熵损失
        loss = -1. / self.num * (
                    torch.matmul(self.labels.t(), torch.log(self.predicts)) + torch.matmul((1 - self.labels.t()),
                                                                                           torch.log(
                                                                                               1 - self.predicts)))
        loss = torch.squeeze(loss, axis=1)
        return loss
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


import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def make_moons(n_samples=1000, shuffle=True, noise=None):
    """
    生成带噪音的弯月形状数据
    输入：
        - n_samples：数据量大小，数据类型为int
        - shuffle：是否打乱数据，数据类型为bool
        - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
    输出：
        - X：特征数据，shape=[n_samples,2]
        - y：标签数据, shape=[n_samples]
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    # 采集第1类数据，特征为(x,y)
    # 使用'torch.linspace'在0到pi上均匀取n_samples_out个值
    # 使用'torch.cos'计算上述取值的余弦值作为特征1，使用'torch.sin'计算上述取值的正弦值作为特征2
    outer_circ_x = torch.cos(torch.linspace(0, math.pi, n_samples_out))
    outer_circ_y = torch.sin(torch.linspace(0, math.pi, n_samples_out))
    inner_circ_x = 1 - torch.cos(torch.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_samples_in))
    print('外弯月特征x的形状:', outer_circ_x.shape, '外弯月特征y的形状:', outer_circ_y.shape)
    print('内弯月特征x的形状:', inner_circ_x.shape, '内弯月特征y的形状:', inner_circ_y.shape)

    # 使用'torch.cat'将两类数据的特征1和特征2分别沿维度0拼接在一起，得到全部特征1和特征2
    # 使用'torch.stack'将两类特征沿维度1堆叠在一起
    X = torch.stack(
        [torch.cat([outer_circ_x, inner_circ_x]),
         torch.cat([outer_circ_y, inner_circ_y])],
        axis=1
    )

    print('拼接后的形状:', torch.cat([outer_circ_x, inner_circ_x]).shape)
    print('X的形状:', X.shape)

    # 使用'torch.zeros'将第一类数据的标签全部设置为0
    # 使用'torch.ones'将第二类数据的标签全部设置为1
    y = torch.cat(
        [torch.zeros(size=[n_samples_out]), torch.ones(size=[n_samples_in])]
    )

    print('y的形状:', y.shape)

    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        # 使用'torch.randperm'生成一个数值在0到X.shape[0]，随机排列的一维Tensor作为索引值，用于打乱数据
        print(X.shape[0])

        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    # 如果noise不为None，则给特征值加入噪声
    if noise is not None:
        # 使用'torch.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
        print(noise)
        X += torch.normal(mean=0.0, std=noise, size=X.shape)

    return X, y

# 采样1000个样本
n_samples = 1000
X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.2)

# 可视化生成的数据集，不同颜色代表不同类别
plt.figure(figsize=(5,5))
plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
plt.xlim(-3,4)
plt.ylim(-3,4)
plt.savefig('线性数据集可视化.pdf')
plt.show()

num_train = 640
num_dev = 160
num_test = 200
X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]
y_train = y_train.reshape([-1,1])
y_dev = y_dev.reshape([-1,1])
y_test = y_test.reshape([-1,1])
# 打印X_train和y_train的维度
print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)
# 固定随机种子，保持每次运行结果一致
torch.manual_seed(102)
# 特征维度
input_dim = 2
# 学习率
lr = 0.3
# 实例化模型
model = ModelLR(input_dim=input_dim)
# 指定优化器
optimizer = SimpleBatchGD(init_lr=lr, model=model)
# 指定损失函数
loss_fn = BinaryCrossEntropyLoss()
# 指定评价方式
metric = accuracy

# 实例化RunnerV2类，并传入训练配置
runner = RunnerV2(model, optimizer, metric, loss_fn)
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=1000, log_epochs=200, save_path="best_model.pdparams")

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

plot(runner,fig_name='linear-acc.pdf')

def decision_boundary(w, b, x1):
    w1, w2 = w.flatten()  # 将权重转换为一维数组
    x2 = (- w1 * x1 - b) / w2  # 计算对应的 x2 值
    return x2
# 绘制训练集上的决策边界
plt.figure(figsize=(5, 5))
# 绘制训练数据
plt.scatter(X_train[:, 0].tolist(), X_train[:, 1].tolist(), marker='*', c=y_train.tolist(), label='Training Data')

# 获取模型参数
w = model.params['w'].detach().numpy()  # 转换为numpy数组
b = model.params['b'].detach().numpy()  # 转换为numpy数组

# 生成x1的范围
x1 = torch.linspace(-2, 3, 1000).detach().numpy()  # 转换为numpy数组
x2 = decision_boundary(w, b, x1)  # 计算决策边界的x2值

# 绘制决策边界
plt.plot(x1, x2, color="red", label='Decision Boundary')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary on Training Data')
plt.legend()
plt.show()


#===========模型评价======================
score, loss = runner.evaluate([X_test, y_test])
print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))

plt.figure(figsize=(5,5))
# 绘制原始数据
plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), marker='*', c=y.tolist())

w = model.params['w']
b = model.params['b']
x1 = torch.linspace(-2, 3, 1000)
x2 = decision_boundary(w, b, x1)
# 绘制决策边界
plt.plot(x1.tolist(), x2.tolist(), color="red")
plt.show()