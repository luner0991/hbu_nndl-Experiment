import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
import torch
import  torch.nn as nn
from PIL import Image
# 加载数据集
train_set, dev_set, test_set = json.load(gzip.open(r'D:\i don`t want to open\深度学习-实验\实验11-卷积神经网络（2）-LeNet-Mnisit\mnist.json.gz', 'rb'))

# 数据预处理 将图像的尺寸修改为32*32，转换为tensor形式。并且将输入图像分布改为均值为，标准差为1的正态分布
transforms = transforms.Compose(
    [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])


# 数据集类的定义 定义MNIST_dataset类，继承dataset类
class MNIST_dataset(Dataset):
    # 初始化数据集，接收一个数据集dataset,转换操作transform  测试操作mode='train'
    def __init__(self, dataset, transforms, mode='train'):
        self.mode = mode
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        # 获取图像和标签
        image, label = self.dataset[0][idx], self.dataset[1][idx]

        # 将图像转换为float32类型
        image = np.array(image).astype('float32')
        label = int(label)

        # 重塑图像为28x28的大小（假设原图是784维向量）
        image = np.reshape(image, [28, 28])

        # 将 NumPy 数组转换为 PIL 图像
        image = Image.fromarray(image.astype('uint8'), mode='L')

        # 如果需要转换为灰度图像（如果原图是彩色图像）
        #image = image.convert('L')  # 这里其实对 MNIST 图像没必要，因为原图已是灰度图像

        # 应用转换操作
        image = self.transforms(image)

        return image, label

    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.dataset[0])


# 加载 mnist 数据集 这些数据集在MNIST_dataset类中被初始化，并用于训练、测试和开发模型
train_dataset = MNIST_dataset(dataset=train_set, transforms=transforms, mode='train')
test_dataset = MNIST_dataset(dataset=test_set, transforms=transforms, mode='test')
dev_dataset = MNIST_dataset(dataset=dev_set, transforms=transforms, mode='dev')



class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(LeNet, self).__init__()
        # 卷积层：输出通道数为6，卷积核大小为5×5
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        # 汇聚层：汇聚窗口为2×2，步长为2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # 卷积层：输入通道数为6，输出通道数为16，卷积核大小为5×5，步长为1
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # 汇聚层：汇聚窗口为2×2，步长为2
        self.pool4 = nn.AvgPool2d(2, stride=2)
        # 卷积层：输入通道数为16，输出通道数为120，卷积核大小为5×5
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        # 全连接层：输入神经元为120，输出神经元为84
        self.linear6 = nn.Linear(120, 84)
        # 全连接层：输入神经元为84，输出神经元为类别数
        self.linear7 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1：卷积层+激活函数
        output = F.relu(self.conv1(x))
        # S2：汇聚层
        output = self.pool2(output)
        # C3：卷积层+激活函数
        output = F.relu(self.conv3(output))
        # S4：汇聚层
        output = self.pool4(output)
        # C5：卷积层+激活函数
        output = F.relu(self.conv5(output))
        # 输入层将数据拉平[B,C,H,W] -> [B,CxHxW]
        output = torch.squeeze(output, dim=3)
        output = torch.squeeze(output, dim=2)
        # F6：全连接层
        output = F.relu(self.linear6(output))
        # F7：全连接层
        output = self.linear7(output)
        return output



# 这里用np.random创建一个随机数组作为输入数据
inputs = np.random.randn(*[1, 1, 32, 32])
inputs = inputs.astype('float32')
model = LeNet(in_channels=1, num_classes=10)
c = []
for a, b in model.named_children():
    c.append(a)
print(c)
x = torch.tensor(inputs)
for a, item in model.named_children():
    try:
        x = item(x)
    except:
        x = torch.reshape(x, [x.shape[0], -1])
        x = item(x)
    print(a, x.shape, sep=' ', end=' ')
    for name, value in item.named_parameters():
        print(value.shape, end=' ')
    print()

import time
x = torch.tensor(inputs)
# 创建LeNet类的实例，指定模型名称和分类的类别数目
model = LeNet(in_channels=1, num_classes=10)
# 计算LeNet类的运算速度
model_time = 0
for i in range(60):
    strat_time = time.time()
    out = model(x)
    end_time = time.time()
    # 预热10次运算，不计入最终速度统计
    if i < 10:
        continue
    model_time += (end_time - strat_time)
avg_model_time = model_time / 50
print('LeNet speed:', avg_model_time, 's')
# 计算参数量
# 计算参数量
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model= LeNet(in_channels=1, num_classes=10).to(device)
summary(model, (1,32, 32))
from thop import profile
# 创建一个假输入并将其移动到与模型相同的设备
dummy_input = torch.randn(1, 1, 32, 32).to(device)
# 计算 FLOPS 和参数量
flops, params = profile(model, (dummy_input,))
print(flops,params)
class RunnerV3(object):
    def __init__(self, model, optimizer, loss_fn, metric, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric  # 只用于计算评价指标

        # 记录训练过程中的评价指标变化情况
        self.dev_scores = []

        # 记录训练过程中的损失函数变化情况
        self.train_epoch_losses = []  # 一个epoch记录一次loss
        self.train_step_losses = []  # 一个step记录一次loss
        self.dev_losses = []

        # 记录全局最优指标
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, **kwargs):
        # 将模型切换为训练模式
        self.model.train()

        # 传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        # 传入log打印频率，如果没有传入值则默认为100
        log_steps = kwargs.get("log_steps", 100)
        # 评价频率
        eval_steps = kwargs.get("eval_steps", 0)

        # 传入模型保存路径，如果没有传入值则默认为"best_model.pdparams"
        save_path = kwargs.get("save_path", "best_model.pdparams")

        custom_print_log = kwargs.get("custom_print_log", None)

        # 训练总的步数
        num_training_steps = num_epochs * len(train_loader)

        if eval_steps:
            if self.metric is None:
                raise RuntimeError('Error: Metric can not be None!')
            if dev_loader is None:
                raise RuntimeError('Error: dev_loader can not be None!')

        # 运行的step数目
        global_step = 0

        # 进行num_epochs轮训练
        for epoch in range(num_epochs):
            # 用于统计训练集的损失
            total_loss = 0
            for step, data in enumerate(train_loader):
                X, y = data
                # 获取模型预测
                logits = self.model(X)
                loss = self.loss_fn(logits, y)  # 默认求mean
                total_loss += loss

                # 训练过程中，每个step的loss进行保存
                self.train_step_losses.append((global_step, loss.item()))

                if log_steps and global_step % log_steps == 0:
                    print(
                        f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")

                # 梯度反向传播，计算每个参数的梯度值
                loss.backward()

                if custom_print_log:
                    custom_print_log(self)

                # 小批量梯度下降进行参数更新
                self.optimizer.step()
                # 梯度归零
                optimizer.zero_grad()

                # 判断是否需要评价
                if eval_steps > 0 and global_step > 0 and \
                        (global_step % eval_steps == 0 or global_step == (num_training_steps - 1)):

                    dev_score, dev_loss = self.evaluate(dev_loader, global_step=global_step)
                    print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}")

                    # 将模型切换为训练模式
                    self.model.train()

                    # 如果当前指标为最优指标，保存该模型
                    if dev_score > self.best_score:
                        self.save_model(save_path)
                        print(
                            f"[Evaluate] best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                        self.best_score = dev_score

                global_step += 1

            # 当前epoch 训练loss累计值
            trn_loss = (total_loss / len(train_loader)).item()
            # epoch粒度的训练loss保存
            self.train_epoch_losses.append(trn_loss)

        print("[Train] Training done!")

    # 模型评估阶段，使用'torch.no_grad()'控制不计算和存储梯度
    @torch.no_grad()
    def evaluate(self, dev_loader, **kwargs):
        assert self.metric is not None

        # 将模型设置为评估模式
        self.model.eval()

        global_step = kwargs.get("global_step", -1)

        # 用于统计训练集的损失
        total_loss = 0

        # 重置评价
        self.metric.reset()

        # 遍历验证集每个批次
        for batch_id, data in enumerate(dev_loader):
            X, y = data

            # 计算模型输出
            logits = self.model(X)

            # 计算损失函数
            loss = self.loss_fn(logits, y).item()
            # 累积损失
            total_loss += loss

            # 累积评价
            self.metric.update(logits, y)

        dev_loss = (total_loss / len(dev_loader))
        dev_score = self.metric.accumulate()

        # 记录验证集loss
        if global_step != -1:
            self.dev_losses.append((global_step, dev_loss))
            self.dev_scores.append(dev_score)

        return dev_score, dev_loss

    # 模型评估阶段，使用'torch.no_grad()'控制不计算和存储梯度
    @torch.no_grad()
    def predict(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)


# Accuracy
class Accuracy():
    def __init__(self):
        """
        输入：
           - is_logist: outputs是logist还是激活后的值
        """

        # 用于统计正确的样本个数
        self.num_correct = 0
        # 用于统计样本的总数
        self.num_count = 0

        self.is_logist = True

    def update(self, outputs, labels):
        """
        输入：
           - outputs: 预测值, shape=[N,class_num]
           - labels: 标签值, shape=[N,1]
        """

        # 判断是二分类任务还是多分类任务，shape[1]=1时为二分类任务，shape[1]>1时为多分类任务
        if outputs.shape[1] == 1:  # 二分类
            outputs = torch.squeeze(outputs, dim=-1)
            if self.is_logist:
                # logist判断是否大于0
                preds = torch.can_cast((outputs >= 0), dtype=torch.float32)
            else:
                # 如果不是logist，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0
                preds = torch.can_cast((outputs >= 0.5), dtype=torch.float32)
        else:
            # 多分类时，使用'torch.argmax'计算最大元素索引作为类别
            preds = torch.argmax(outputs, dim=1).int()

        # 获取本批数据中预测正确的样本个数
        labels = torch.squeeze(labels, dim=-1)
        # batch_correct = torch.sum(torch.tensor(preds == labels, dtype=torch.float32)).numpy()
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


# 进行训练
import torch.optim as opti
from torch.utils.data import DataLoader

# 学习率大小
lr = 0.1
# 批次大小
batch_size = 64

# 创建三个数据加载器，分别用于训练、开发和测试数据集 shuffle=True表示在每个epoch开始时对数据进行随机打乱，防止过拟合
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# 定义LeNet网络
model = LeNet(in_channels=1, num_classes=10)
# 定义优化器 优化器的学习率设置为0.2
optimizer = opti.SGD(model.parameters(), 0.2)
# 定义损失函数 使用交叉熵损失函数
loss_fn = F.cross_entropy
# 定义评价指标-这里使用的是准确率
metric = Accuracy()
# 实例化 RunnerV3 类，并传入模型、优化器、损失函数和评价指标
runner = RunnerV3(model, optimizer, loss_fn, metric)
# 启动训练，设置每15步记录一次日志  每15步评估一次模型性能
log_steps = 15
eval_steps = 15
# 训练模型6个epoch,并保存最好的模型参数
runner.train(train_loader, dev_loader, num_epochs=6, log_steps=log_steps,
             eval_steps=eval_steps, save_path="best_model.pdparams")

runner.load_model('best_model.pdparams')
# 加载最优模型
runner.load_model('best_model.pdparams')
# 模型评价
score, loss = runner.evaluate(test_loader)
print("[Test] accuracy/loss: {:.4f}/{:.4f}".format(score, loss))