'''
@function: 基于LeNet识别MNIST数据集
@Author: lxy
@date: 2024/11/14
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义超参数
input_size = 28  # 图像的总尺寸28*28
num_classes = 10  # 标签的种类数
num_epochs = 14  # 训练的总循环周期
batch_size = 64  # 一个撮（批次）的大小，64张图片

# 训练集
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class LetNet(nn.Module):
    def __init__(self):
        super(LetNet, self).__init__()
        # 第一层卷积：输入 (1, 28, 28) -> 输出 (6, 12, 12)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入通道数：1（灰度图）
                out_channels=6,  # 输出通道数：6
                kernel_size=5,  # 卷积核大小：5x5
                stride=1,  # 步长：1
            ),  # 输出特征图 (6, 24, 24)
            nn.BatchNorm2d(6),  # 批标准化
            nn.ReLU(),  # ReLU 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化操作（2x2区域）-> 输出 (6, 12, 12)
        )
        # 第二层卷积：输入 (6, 12, 12) -> 输出 (16, 4, 4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),  # 卷积核大小：5x5，步长：1 -> 输出 (16, 8, 8)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 池化操作（2x2区域）-> 输出 (16, 4, 4)
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Linear(120 , 84),  # 120x1x1 展平后输入到全连接层 -> 84个输出
            nn.Linear(84, 10)  # 最后一层输出10个类别
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)  # 展平操作，保持batch_size，展平特征图
        output = self.fc(x)  # 通过全连接层进行分类
        return output


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    train_rights = []
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        right = accuracy(output, target)
        train_rights.append(right)
        total_loss += loss.item()
    # 计算训练准确率
    train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
    avg_loss = total_loss / len(train_loader)
    train_acc = 100. * train_r[0] / train_r[1]
    return avg_loss, train_acc


def evaluate(model, criterion, test_loader):
    model.eval()
    val_rights = []
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            right = accuracy(output, target)
            val_rights.append(right)

    # 计算测试准确率
    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
    avg_loss = total_loss / len(test_loader)
    val_acc = 100. * val_r[0] / val_r[1]

    return avg_loss, val_acc


# 实例化模型
net = LetNet()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练和测试过程
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    # 训练阶段
    train_loss, train_acc = train_one_epoch(net, criterion, optimizer, train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # 测试阶段
    test_loss, test_acc = evaluate(net, criterion, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"训练集损失: {train_loss:.4f}, 训练集准确率: {train_acc:.2f}%")
    print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.2f}%")

# 可视化训练过程中的损失和准确率
epochs = np.arange(1, num_epochs + 1)

# 绘制损失图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='loss_train')
plt.plot(epochs, test_losses, label='loss_test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('loos')

# 绘制准确率图
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Accuracy_train')
plt.plot(epochs, test_accuracies, label='Accuracy_test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()
