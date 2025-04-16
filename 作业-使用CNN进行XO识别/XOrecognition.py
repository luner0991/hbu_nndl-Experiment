'''
@Author :lxy
@function : XOrecognition based in CNN
@date :2024/10/26
'''
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义数据的预处理操作，将图片转换为灰度图并归一化为Tensor
transforms = transforms.Compose([
    transforms.ToTensor(),      # 将图片数据转换为Tensor类型，数值归一化到[0, 1]范围
    transforms.Grayscale(1)     # 将图片转换为单通道的灰度图
])

# 定义训练集和测试集路径
path_train = r'train_data'
path_test = r'test_data'

# 加载训练和测试数据集，应用数据预处理操作
data_train = datasets.ImageFolder(path_train, transform=transforms)
data_test = datasets.ImageFolder(path_test, transform=transforms)

# 打印训练集和测试集的大小
print("size of train_data:", len(data_train))
print("size of test_data:", len(data_test))

# 使用DataLoader将数据集分成小批量进行加载，batch_size=64表示每次加载64个样本
train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
test_loader = DataLoader(data_test, batch_size=64, shuffle=True)

# 打印训练集样本的形状
for i, data in enumerate(train_loader):
    images, labels = data
    print(images.shape)  # 打印输入图像的形状
    print(labels.shape)  # 打印标签的形状
    break  # 只打印第一批数据，避免输出过多

# 打印测试集样本的形状
for i, data in enumerate(test_loader):
    images, labels = data
    print(images.shape)
    print(labels.shape)
    break

# 定义卷积神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积，输入为灰度图像的单通道，输出为9个特征通道，卷积核大小为3x3，步长为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3)
        # 最大池化层，池化窗口为2x2，步长为2
        self.maxpool = nn.MaxPool2d(2, 2)
        # 第二层卷积，输入为9个通道，输出为5个特征通道，卷积核大小为3x3，步长为1
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=5, kernel_size=3 )
        # 激活函数
        self.relu = nn.ReLU()

        '''
                 # 第一层卷积输出特征图尺寸： (116 - 3) / 1 + 1 = 114
                 池化：114/2=57
                 # 第二层卷积输出特征图尺寸： (57 - 3) / 1 + 1 = 55
                 池化：55/2=27.5上取整为27
                 进入全连接输入大小为27 * 27 * 5
                '''
        # 全连接层1，输入大小为27*27*5（根据卷积和池化操作后的特征图尺寸），输出大小为1200
        self.fc1 = nn.Linear(27 * 27 * 5, 1200)
        # 全连接层2，输入大小为1200，输出大小为64
        self.fc2 = nn.Linear(1200, 64)
        # 输出层，输入大小为64，输出大小为2（用于二分类）x/o
        self.fc3 = nn.Linear(64, 2)

    # 前向传播过程
    def forward(self, x):
        # 第一层卷积+激活+池化
        x = self.maxpool(self.relu(self.conv1(x)))
        # 第二层卷积+激活+池化
        x = self.maxpool(self.relu(self.conv2(x)))
        # 展平操作，将多维特征图展平为一维
        x = x.view(-1, 27 * 27 * 5)
        # 第一个全连接层+激活
        x = self.relu(self.fc1(x))
        # 第二个全连接层+激活
        x = self.relu(self.fc2(x))
        # 输出层，不加激活函数，用于分类任务
        x = self.fc3(x)
        return x

# 初始化网络、损失函数和优化器
model = Net()  # 实例化网络模型
criterion = torch.nn.CrossEntropyLoss()  # 损失函数为交叉熵，用于分类任务
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 优化器为随机梯度下降法，学习率为0.1

# 训练网络
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0  # 用于记录损失
    for i, data in enumerate(train_loader):
        images, label = data
        out = model(images)  # 前向传播，得到模型输出
        loss = criterion(out, label)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            # 每10个小批量数据打印一次损失
            print('Epoch[%d/%5d], loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0  # 重置损失值
print('训练结束')

# 保存模型的状态字典，包含模型的权重和偏置
torch.save(model.state_dict(), 'bestmodel')

# =======================加载并测试单张图片==========================
model_load = Net()  # 重新创建模型实例
model.load_state_dict(torch.load('bestmodel'))
model_load.eval()  # 切换到评估模式

# 从测试集中取出一张图片
images, labels = next(iter(test_loader))
print("labels[0] truth:\t", labels[0].item())  # 打印真实标签
x = images[0].unsqueeze(0)  # 增加一个batch维度以匹配模型输入
with torch.no_grad():  # 禁用梯度计算，加快推理速度
    output = model_load(x)
    _, predicted = torch.max(output, 1)  # 获取预测类别
    print("labels[0] predict:\t", predicted.item())  # 打印预测标签

# 显示测试图像
img = images[0].data.squeeze().numpy()  # 将图像转换为numpy数组并去除batch维度
plt.imshow(img, cmap='gray')  # 显示为灰度图
plt.show()

# ============================测试模型在整个测试集上的准确率==========
correct = 0  # 记录正确预测数
total = 0  # 记录总样本数
# 禁用梯度计算
with torch.no_grad():
    # 遍历测试数据加载器，取出每批数据
    for data in test_loader:
        images, labels = data  # 获取输入图像和真实标签
        outputs = model_load(images)  # 前向传播，计算模型输出
        _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的预测类别，取输出的最大值索引作为预测结果
        total += labels.size(0)  # 记录总样本数
        correct += (predicted == labels).sum().item()  # 统计预测正确的样本数

accuracy = 100. * correct / total
print(f'网络在整个测试集上的准确率: {accuracy:f}%')
