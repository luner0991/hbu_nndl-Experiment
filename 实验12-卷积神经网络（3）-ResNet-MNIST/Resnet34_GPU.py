'''
task：实现resnet34残差网络
date: 2024/11/12
'''
import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,outchannel_list,stride):
        super(ResidualBlock, self).__init__()
        # 第一个卷积操作
        self.conv1 = nn.Conv2d(in_channels,outchannel_list[0],3,stride,1,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel_list[0])
        self.relu = nn.ReLU(True)
        # 第二个卷积操作
        self.conv2 = nn.Conv2d(outchannel_list[0],outchannel_list[1],3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel_list[1])
        self.shortcut = nn.Sequential()
        if in_channels!=outchannel_list[-1] or stride!=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,outchannel_list[-1],1,stride,0,bias=False),
                nn.BatchNorm2d(outchannel_list[-1])
            )
    def forward(self,input):
        initial = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        initial = self.shortcut(initial)

        output = output+ initial
        return output
class Resnet34(nn.Module):
    def __init__(self,sort_classes):
        super(Resnet34, self).__init__()
        # input = 3*224*224 output=64*112*112
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3,2,1)
        )
         # 64*56*56
        self.stage2 = nn.Sequential(
            ResidualBlock(64,[64,64],1),
            ResidualBlock(64,[64,64],1),
            ResidualBlock(64,[64,64],1),
        )
        # 128* 28 *28
        self.stage3 = nn.Sequential(
            ResidualBlock(64,[128,128],2),
            ResidualBlock(128,[128,128],1),
            ResidualBlock(128,[128,128],1),
            ResidualBlock(128,[128,128],1),
        )
        # 256*14*14
        self.stage4 = nn.Sequential(
            ResidualBlock(128,[256,256],2),
            ResidualBlock(256,[256,256],1),
            ResidualBlock(256,[256,256],1),
            ResidualBlock(256,[256,256],1),
            ResidualBlock(256,[256,256],1),
            ResidualBlock(256,[256,256],1),
            ResidualBlock(256,[256,256],1),
        )
        # 512 * 7 * 7
        self.stage5 = nn.Sequential(
            ResidualBlock(256,[512,512],2),
            ResidualBlock(512,[512,512],1),
            ResidualBlock(512,[512,512],1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,sort_classes)
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

model = Resnet34(10)
print(model)
# 导入torchsummary
from torchsummary import summary
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型加载到设备上
model = Resnet34(10).to(device)

# 使用summary函数时，也需要确保输入数据位于相同设备上
summary(model, input_size=(3, 224, 224), device=str(device))

# 使用summary函数
summary(model, input_size=(3, 224, 224))

# 模型训练 ==================
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 定义transfroms
transforms = transforms.Compose([
    transforms.Resize((224,224)), # 重新设置图片尺寸
    transforms.ToTensor(), # 转换为tensor格式
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])] # 归一化
)
# 创建数据集合
TrainSet = torchvision.datasets.CIFAR10('../data', train=True, transform=transforms, download=True)
TestSet = torchvision.datasets.CIFAR10('../data', train=False, transform=transforms, download=True)
TrainLoader = DataLoader(TrainSet,batch_size=8,shuffle=True)
TestLoader = DataLoader(TestSet,batch_size=8,shuffle=False)

# 定义优化器
import torch.optim as optim
optimizer = optim.SGD(params=model.parameters(),lr=0.001,momentum=0.9)

# 定义损失函数
CrossEntroy = nn.CrossEntropyLoss() # 图片分类任务一般使用交叉熵损失函数

dataiter = iter(TrainLoader)

images,labels = dataiter.__next__()

def imshow(img):
    # 反标准化
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.numpy()
    img = (img * 255).astype(np.uint8)  # 将图像值缩放到[0, 255]范围，并转为整数类型
    # 需要更改通道位置，Pytorch读入以后默认为Depth,Width,Height
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
imshow(torchvision.utils.make_grid(images))

# 开始训练
epochs = 1 # 训练所需的轮数，自行设置
for epoch in range(epochs):
    running_loss = 0
    loss_list = []
    for i,data in enumerate(dataiter,0):
        # 将输入数据和标签转移到 GPU 上
        images, labels = images.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        outputs = model(images)
        loss = CrossEntroy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 5 == 0 and i>=5:
            print('第{0}轮,第{1}次迭代以后的loss:{2}'.format(epoch + 1, i, running_loss))
            loss_list.append(running_loss/5)
            running_loss = 0.0
print("训练完成!")
print("running_loss变化:",loss_list)

# 类别名称
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

save_path = './cifar_ResNet18.pth'
torch.save(model.state_dict().path)



