import torch
import torch.nn as nn
import torch.nn.functional as F
from Runner2 import RunnerV2  # 导入 RunnerV2 类
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

#===================生成数据===============================
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def make_multiclass_classification(n_samples=100, n_features=2, n_classes=3, shuffle=True, noise=0.1):
    """
    生成带噪音的多类别数据
    输入：
        - n_samples：数据量大小，数据类型为int
        - n_features：特征数量，数据类型为int
        - shuffle：是否打乱数据，数据类型为bool
        - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
    输出：
        - X：特征数据，shape=[n_samples,2]
        - y：标签数据, shape=[n_samples,1]
    """
    # 计算每个类别的样本数量
    n_samples_per_class = [int(n_samples / n_classes) for k in range(n_classes)]
    for i in range(n_samples - sum(n_samples_per_class)):
        n_samples_per_class[i % n_classes] += 1
    # 将特征和标签初始化为0
    X = torch.zeros([n_samples, n_features])
    y = torch.zeros([n_samples], dtype=torch.int32)
    # 随机生成3个簇中心作为类别中心
    centroids = torch.randperm(2 ** n_features)[:n_classes]
    centroids_bin = np.unpackbits(centroids.numpy().astype('uint8')).reshape((-1, 8))[:, -n_features:]
    centroids = torch.tensor(centroids_bin, dtype=torch.float32)
    # 控制簇中心的分离程度
    centroids = 1.5 * centroids - 1
    # 随机生成特征值
    X[:, :n_features] = torch.randn(size=[n_samples, n_features])

    stop = 0
    # 将每个类的特征值控制在簇中心附近
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_class[k]
        # 指定标签值
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_features]
        # 控制每个类别特征值的分散程度
        A = 2 * torch.rand(size=[n_features, n_features]) - 1
        X_k[...] = torch.matmul(X_k, A)
        X_k += centroid
        X[start:stop, :n_features] = X_k

    # 如果noise不为None，则给特征加入噪声
    if noise > 0.0:
        # 生成noise掩膜，用来指定给那些样本加入噪声
        noise_mask = torch.rand([n_samples]) < noise
        for i in range(len(noise_mask)):
            if noise_mask[i]:
                # 给加噪声的样本随机赋标签值
                y[i] = torch.randint(0,n_classes, (1,),dtype=torch.int32)
    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    return X, y
# 固定随机种子，保持每次运行结果一致
torch.manual_seed(102)
# 采样1000个样本
n_samples = 1000
X, y = make_multiclass_classification(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)

# 可视化生产的数据集，不同颜色代表不同类别
plt.figure(figsize=(5,5))
plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
plt.savefig('linear-dataset-vis2.pdf')
plt.show()
num_train = 640
num_dev = 160
num_test = 200

X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

# 打印X_train和y_train的维度
print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)
# 固定随机种子，保持每次运行结果一致
torch.manual_seed(102)

# 特征维度
input_dim = 2
# 类别数
output_dim = 3
# 学习率
lr = 0.1

# 实例化模型
model = ModelSR(input_dim=input_dim, output_dim=output_dim)
# 指定优化器
optimizer = SimpleBatchGD(init_lr=lr, model=model)
# 指定损失函数
loss_fn = MultiCrossEntropyLoss()
# 指定评价方式
metric = accuracy
# 实例化RunnerV2类
runner = RunnerV2(model, optimizer, metric, loss_fn)

# 模型训练
runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=500, log_eopchs=50, eval_epochs=1,
             save_path="best_model.pdparams")


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

#==================模型评价================================
score, loss = runner.evaluate([X_test, y_test])
print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))
# 均匀生成40000个数据点
x1, x2 = torch.meshgrid(torch.linspace(-3.5, 2, 200), torch.linspace(-4.5, 3.5, 200),indexing='ij')
x = torch.stack([torch.flatten(x1), torch.flatten(x2)], dim=1)
# 预测对应类别
y = runner.predict(x)
y = torch.argmax(y, dim=1)
# 绘制类别区域
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(x[:, 0].tolist(), x[:, 1].tolist(), c=y.tolist(), cmap=plt.cm.Spectral)

torch.manual_seed(102)
n_samples = 1000
X, y = make_multiclass_classification(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)

plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), marker='*', c=y.tolist())
plt.show()
