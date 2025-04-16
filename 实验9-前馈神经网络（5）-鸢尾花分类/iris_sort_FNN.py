'''
@Author: lxy
@function: Classification of the Iris Dataset Based on FNN
@date: 2024/11/1
'''
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_iris
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn.init import normal_,constant_
from Runner3 import RunnerV3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class IrisDataset(Dataset):
    # mode 用于区分数据集类型，训练集train,验证集dev
    def __init__(self, mode='train', num_train=120, num_dev=15):
        super(IrisDataset, self).__init__()
        # 调用名为load_data的函数，此函数负责加载鸢尾花数据集
        # shuffle=True表示在加载数据时会被随机打乱，确保每个epoch的顺序是随机的
        X, y = load_data(shuffle=True)
        # ========================分割数据集=============================
        # 作为训练集
        if mode == 'train':
            self.X, self.y = X[:num_train], y[:num_train]
        # 作为验证集
        elif mode == 'dev':
            self.X, self.y = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
        # 去剩余样本作为测试集
        else:
            self.X, self.y = X[num_train + num_dev:], y[num_train + num_dev:]
    # 从数据集中获取一个样本，返回一个样本的输入X和标签y
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    # 返回数据集标签的数量(即数据集的大小)
    def __len__(self):
        return len(self.y)

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
    iris = load_iris()
    X = np.array(iris.data, dtype=np.float32)
    # 注意int 64
    y = np.array(iris.target, dtype=np.int64)
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
# 设置随机种子以保证结果的可重复性
torch.random.manual_seed(12)
# 创建了训练集、验证集和测试集的实例
train_dataset = IrisDataset(mode='train')
dev_dataset = IrisDataset(mode='dev')
test_dataset = IrisDataset(mode='test')
# 打印训练集长度
print("length of train set: ", len(train_dataset))

# ===========================用DataLoader进行封装=============================

# 将数据集封装成DataLoader，方便批量加载和打乱数据
batch_size =16 #设置批次大小
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

#  ================================模型构建====================================
# 模型构建
# 实现一个两层前馈神经网络
class IrisSort(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IrisSort, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        normal_(self.fc1.weight, mean=0, std=0.01)
        constant_(self.fc1.bias, val=1.0)
        self.fc2 = nn.Linear(hidden_size, output_size)
        normal_(self.fc2.weight, mean=0, std=0.01)
        constant_(self.fc2.bias, val=1.0)
        self.act = torch.sigmoid

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.act(outputs)
        outputs = self.fc2(outputs)
        return outputs


fnn_model = IrisSort(input_size=4, hidden_size=6, output_size=3)


# Accuracy代码
class Accuracy(object):
    def __init__(self, is_logist=True):
        # 用于统计正确的样本个数
        self.num_correct = 0
        # 用于统计样本的总数
        self.num_count = 0
        self.is_logist = is_logist

    def update(self, outputs, labels):
        # 判断是二分类任务还是多分类任务，shape[1]=1时为二分类任务，shape[1]>1时为多分类任务
        if outputs.shape[1] == 1:  # 二分类
            outputs = torch.squeeze(outputs, axis=-1)
            if self.is_logist:
                # logist判断是否大于0
                preds = (outputs >= 0).to(torch.float32)
            else:
                # 如果不是logist，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0
                preds = (outputs >= 0.5).to(torch.float32)
        else:
            # 多分类时，使用'torch.argmax'计算最大元素索引作为类别
            preds = torch.argmax(outputs, dim=1)
        labels = labels.long()  # 确保标签是 Long 类型
        # 获取本批数据中预测正确的样本个数
        labels = torch.squeeze(labels, axis=-1)
        batch_correct = torch.sum((preds == labels).clone().detach()).numpy()
        batch_count = len(labels)

        # 更新num_correct 和 num_count
        self.num_correct += batch_correct
        self.num_count += batch_count

    def accumulate(self):
        # 使用累计的数据，计算总的指标
        if self.num_count == 0:
            return 0
        return self.num_correct / self.num_count

    def reset(self):
        # 重置正确的数目和总数
        self.num_correct = 0
        self.num_count = 0

    def name(self):
        return "Accuracy"

# =====================================模型训练===============================
lr = 0.2  # 定义学习率
# 定义网络
model = fnn_model  # 模型fnn_model已在上面代码定义
# 定义优化器SGD  随机梯度下降优化器，将模型参数传给优化器
optimizer = SGD(model.parameters(), lr=lr)
# 定义损失函数交叉熵
loss_fn = F.cross_entropy

# 定义评价指标 准确率
metric = Accuracy(is_logist=True)
runner = RunnerV3(model, optimizer, metric,loss_fn,)

# 启动训练 训练轮数为150轮，每隔100步记录一次，每隔50步进行一次评估
log_steps = 100
eval_steps = 50
runner.train(train_loader, dev_loader,
             num_epochs=150, log_steps=log_steps, eval_steps=eval_steps,
             save_path="best_model.pdparams")
# ==============可视化===================
# 绘制训练集和验证集的损失变化以及验证集上的准确率变化曲线
def plot_training_loss_acc(runner, fig_name,
                           fig_size=(16, 6),
                           sample_step=20,  # 取数据的步长，每隔20取一个
                           loss_legend_loc="upper right",
                           acc_legend_loc="lower right",
                           train_color="#8E004D",
                           dev_color='#E20079',
                           fontsize='x-large',
                           train_linestyle="--",
                           dev_linestyle='-',
                           train_linewidth=2,
                           dev_linewidth=2):
    plt.figure(figsize=fig_size)

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    train_items = runner.train_step_losses[::sample_step]
    train_steps = [x[0] for x in train_items]
    train_losses = [x[1] for x in train_items]

    plt.plot(train_steps, train_losses, color=train_color, linestyle=train_linestyle,
             linewidth=train_linewidth, label="Train loss", marker='o', markersize=4)

    if len(runner.dev_losses) > 0:
        dev_steps = [x[0] for x in runner.dev_losses]
        dev_losses = [x[1] for x in runner.dev_losses]
        plt.plot(dev_steps, dev_losses, color=dev_color, linestyle=dev_linestyle,
                 linewidth=dev_linewidth, label="Dev loss", marker='s', markersize=4)

    plt.title("Training and Validation Loss", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Step", fontsize=fontsize)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc=loss_legend_loc, fontsize='x-large')

    # 绘制准确率曲线
    if len(runner.dev_scores) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(dev_steps, runner.dev_scores,
                 color=dev_color, linestyle=dev_linestyle,
                 linewidth=dev_linewidth, label="Dev Accuracy", marker='^', markersize=4)

        plt.title("Validation Accuracy", fontsize=fontsize)
        plt.ylabel("Accuracy", fontsize=fontsize)
        plt.xlabel("Step", fontsize=fontsize)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc=acc_legend_loc, fontsize='x-large')

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(fig_name)  # 保存图像
    plt.show()

plot_training_loss_acc(runner, 'fw-loss.pdf')  # 调用绘制图形的函数

# ==================模型评价==================
# 加载最优模型
runner.load_model('best_model.pdparams')
# 模型评价
score, loss = runner.evaluate(test_loader)
print("[Test] accuracy/loss: {:.4f}/{:.4f}".format(score, loss))

# ============ 模型预测 ====================
# 获取测试集中第一条数据
X ,label = next(iter(test_loader))
logits = runner.predict(X)
pred_class = torch.argmax(logits[0]).numpy()
# 输出真实类别与预测类别
print("The true category is {} and the predicted category is {}".format(label[0], pred_class))