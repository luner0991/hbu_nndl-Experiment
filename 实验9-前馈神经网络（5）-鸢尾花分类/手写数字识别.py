import torch#导入库，提供张量操作和神经网络的构建功能
import torch.nn as nn#导入神经网络模块
import torch.nn.functional as F#导入包含各种激活函数和一些操作的函数的集合
import torch.optim as optim#导入优化器，用于更新神经网络权重
from torchvision import datasets,transforms#导入数据集和数据转换的功能
import torchvision  #导入视觉相关的工具
from torch.autograd import Variable #导入自动求导模块中的变量
from torch.utils.data import DataLoader #导入数据加载器
import cv2 #导入OpenCV库，用于图像处理


#定义LeNet模型
class LeNet(nn.Module): #创建一个名为LeNet的神经网络类，继承自nn.Module
    def __init__(self):  #初始化LetNet类
        super(LeNet, self).__init__()
        #定义了两个卷积层 conv1 conv2，每个卷积层包含卷积操作、ReLU激活函数、最大池化操作
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1, 2),
            #1是输入通道数，表示输入数据的通道数为1；6是输出通道数，表示该层卷积核的数量为6，每个卷积核会输出一个特征图
            #3是卷积核的大小为3*3，1是步长，表示卷积核在每个步长下移动距离为1，2是填充的大小，这里的填充是指在输入图像周围填充0，以控制卷积i操作后特征图的大小
            nn.ReLU(), #激活函数将所有的负值置为0，正值保持不变，用于引入非线性特征
            nn.MaxPool2d(2, 2)#最大池化层用于减少特征图的空间尺寸，保留最重要的特征
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        #定义了三个全连接层（也称线性层）fc1\fc2\fc3、前两个包含线性变换、批标准化和ReLU激活函数，最后一个是输出层
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), #在卷积层之后经过了池化层输出的特征图尺寸为5*5，且有16个通道，120表示这一场的输出大小
            nn.BatchNorm1d(120),
            nn.ReLU()#激活函数层，将该全连接层的输出应用于3非线性变换
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),#加快收敛速度的方法（注：批标准化一般放在全连接层后面，激活函数层的前面）
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10) #最后一个全连接层，84表示输入大小为上一层的输出大小即84，10表示输出大小为10，对应于该神经网络的输出类别数

    #         self.sfx = nn.Softmax()

    def forward(self, x):  #定义了前向传播的过程、描述了数据在网络中的流动路径
        x = self.conv1(x)#将输入x通过第一个卷积层conv1,经过卷积激活和池化操作得到特征图
        x = self.conv2(x)#将上一层的输出x通过第二个卷积层conv2，再次经过卷积激活函数和池化操作，得到更高级的特征图
        #         print(x.shape)
        x = x.view(x.size()[0], -1)#将卷积层的输出展品为一维向量，这一步在进入全连接层前，以将特征图的二维结构转化为一维向量
        x = self.fc1(x)#通过第一个全连接层fc1，对展平后的特征进行线性变换和激活操作
        x = self.fc2(x)#通过第二个全连接层fc2,再次进行线性变换和激活操作
        x = self.fc3(x)#通过最后一个全连接层fc3，得到模型的最后输出，通常用于分类任务
        #         x = self.sfx(x)查看某一层的输出形状
        return x#返回模型的前向传播结果


#设置设备和参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #根据是否有coda支持选择运行设备
batch_size = 64#设置训练所需的批大小 每个训练批次中含有64张手写数字图片
LR = 0.001 #学习率，控制模型参数更新步长的超参数，较小的学习率使得模型收敛更稳定
Momentum = 0.9 #动量参数，用于改善梯度下降的收敛性，有助于加速训练并避免陷入局部最小值

# 下载数据集
train_dataset = datasets.MNIST(root = './data/',  #train_dataset与test_dataset 定义了训练和测试数据集（MNIST手写数字数据集）、包括数据集的位置
                              train=True,   #使用数据集
                              transform = transforms.ToTensor(), #以及数据转换
                              download=False)
test_dataset =datasets.MNIST(root = './data/',#测试数据集
                            train=False,
                            transform=transforms.ToTensor(),
                            download=False)
#建立数据迭代器，分别用于训练和测试
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size, #指定了批量大小
                                          shuffle = True)#和是否打乱数据
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)#测试时通常不会打乱数据

#实现单张图片可视化
# images,labels = next(iter(train_loader))
# img  = torchvision.utils.make_grid(images)
# img = img.numpy().transpose(1,2,0)
# # img.shape
# std = [0.5,0.5,0.5]
# mean = [0.5,0.5,0.5]
# img = img*std +mean
# cv2.imshow('win',img)
# key_pressed = cv2.waitKey(0)


net = LeNet().to(device)#初始化神经网络
criterion = nn.CrossEntropyLoss()#定义损失函数，用于衡量模型输出和实际标签之间的差异
optimizer = optim.SGD(net.parameters(),lr=LR,momentum=Momentum) #使用随机梯度下降SGD作为优化器，对模型的参数进行优化，lr表示学习率，momentum表示动量

epoch = 1  #指定训练的轮数为1
if __name__ == '__main__':
    for epoch in range(epoch):#外层循环是训练的轮数
        sum_loss = 0.0 #初始化损失变量  sum_loss用于累积每个小批次（batch）的损失，以便计算平均损失
        for i, data in enumerate(train_loader):#内层循环遍历每个小批次
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)  #获取输入数据（inputs)和对应的标签（labels)并封装为 Variable对象
            optimizer.zero_grad()#将梯度归零，pytorch默认会累积梯度，而我们在开始每个小批次时需要清零，以免梯度累积影响下一次计算
            outputs = net(inputs)#将数据传入网络进行前向运算，得到模型输出outputs
            loss = criterion(outputs, labels)#得到损失
            loss.backward()#根据损失计算梯度并反向传播
            optimizer.step()#通过梯度做一步参数更新，使得损失逐渐减小

            # print(loss)#累积损失并输出
            sum_loss += loss.item()#累计损失
            if i % 100 == 99:#每处理100个小批次，就输出一次平均损失
                print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))#打印当前轮数，小批次数和平均损失
                sum_loss = 0.0#置零为下一次累积损失左准备

    #验证测试集 测试过程
    net.eval()#将模型变换为测试模式
    correct = 0#初始化模型正确分类的样本数
    total = 0#初始化测试集的总样本数
    for data_test in test_loader:#遍历测试集，每次加载一个小批次的数据
        images, labels = data_test  #获取测试集中输入图像（images)和对应的标签（labels)并封装为 Variable对象
        images, labels = Variable(images), Variable(labels)
        output_test = net(images) #将测试集图像输入到神经网络进行前向传播，得到模型预测输出
        # print("output_test:",output_test.shape)

        _, predicted = torch.max(output_test, 1)#计算准确率，此处的predicted获取的是最大值的下标
        # print("predicted:",predicted.shape)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct1: ",correct)
    print("Test acc: {0}".format(correct.item() / len(test_dataset)))#.cpu().numpy() #使用correct.item获取correct数值，输出测试准确率