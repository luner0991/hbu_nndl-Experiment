'''
@Function: 基于ResNet34实现对MNIST手写数字集的识别
@Author: lxy
@Date: 2021/11/20
'''

import torch
import torch.nn as nn

# 定义残差单元
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,use_residual=True):
        """
        残差单元
               输入：
                - in_channels：输入通道数
                - out_channels：输出通道数
                - stride：残差单元的步长，通过调整残差单元中第一个卷积层的步长来控制
                - use_residual：用于控制是否使用残差连接
        """
        super(ResBlock, self).__init__()
        self.use_residual = use_residual
        # 第一个卷积层，卷积核大小为3×3，可以设置不同输出通道数以及步长
        self.conv1 = nn.Conv2d(in_channels,out_channels[0],3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        # 第二个卷积层，卷积核大小为3×3，不改变输入特征图的形状，步长为1
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1],3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.relu = nn.ReLU(inplace=True)
        # 如果conv2的输出和此残差块的输入数据形状不一致，则use = True
        # 当use = True，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.use = True
        else:
            self.use = False
        if self.use:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels[-1], 1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels[-1])
            )
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_residual:
            if self.use: # 如果为真，对x进行1×1卷积，将形状调整成跟conv2的输出y一致
                identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out
class Resnet34(nn.Module):
    def __init__(self,in_channels,num_classes=10,use_residual=True):
        super(Resnet34, self).__init__()
        # input = 3*224*224 output=64*112*112
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3,2,1)
        )
         # 64*56*56
        self.stage2 = nn.Sequential(
            ResBlock(64,[64,64],1,use_residual=True),
            ResBlock(64,[64,64],1,use_residual=True),
            ResBlock(64,[64,64],1,use_residual=True),
        )
        # 128* 28 *28
        self.stage3 = nn.Sequential(
            ResBlock(64,[128,128],2,use_residual=True),
            ResBlock(128,[128,128],1,use_residual=True),
            ResBlock(128,[128,128],1,use_residual=True),
            ResBlock(128,[128,128],1,use_residual=True),
        )
        # 256*14*14
        self.stage4 = nn.Sequential(
            ResBlock(128,[256,256],2,use_residual=True),
            ResBlock(256,[256,256],1,use_residual=True),
            ResBlock(256,[256,256],1,use_residual=True),
            ResBlock(256,[256,256],1,use_residual=True),
            ResBlock(256,[256,256],1,use_residual=True),
            ResBlock(256,[256,256],1,use_residual=True),
            ResBlock(256,[256,256],1,use_residual=True),
        )
        # 512 * 7 * 7
        self.stage5 = nn.Sequential(
            ResBlock(256,[512,512],2,use_residual=True),
            ResBlock(512,[512,512],1,use_residual=True),
            ResBlock(512,[512,512],1,use_residual=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
    def forward(self,input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.stage5(output)
        output = self.avgpool(output)

        output = torch.flatten(output,1) # 从维度1开始展平
        output = self.fc(output)
        return output





# ===============================数据集========================================
import numpy as np
import json
import gzip
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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
# ==================模型训练==================================
from Runner import RunnerV3,Accuracy,plot
import torch.optim as opti
import torch.nn.functional as F
# 固定随机种子
torch.manual_seed(0)
# 学习率大小
lr = 0.005
# 批次大小
batch_size = 64
# 创建三个数据加载器，分别用于训练、开发和测试数据集 shuffle=True表示在每个epoch开始时对数据进行随机打乱，防止过拟合
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# 定义ResNet网络
model = Resnet34(in_channels=1, num_classes=10,use_residual=True).to(device)
# 定义优化器
optimizer = opti.SGD(model.parameters(), lr)
# 定义损失函数 使用交叉熵损失函数
loss_fn = F.cross_entropy
# 定义评价指标-这里使用的是准确率
metric = Accuracy()
# 实例化 RunnerV3 类，并传入模型、优化器、损失函数和评价指标
runner = RunnerV3(model, optimizer, loss_fn, metric,device)
# 启动训练，设置每15步记录一次日志  每15步评估一次模型性能
log_steps = 15
eval_steps = 15
# 训练模型6个epoch,并保存最好的模型参数
runner.train(train_loader, dev_loader, num_epochs=5, log_steps=log_steps,
             eval_steps=eval_steps, save_path="best_model.pdparams")
# 可视化观察训练集与验证集的Loss变化情况
plot(runner, 'cnn-loss2.pdf')

# 模型评价
# 加载最优模型
runner.load_model('best_model.pdparams')
# 模型评价
score, loss = runner.evaluate(test_loader)
print("[Test] accuracy/loss: {:.4f}/{:.4f}".format(score, loss))












