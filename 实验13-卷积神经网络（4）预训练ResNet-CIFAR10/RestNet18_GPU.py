"""
    @Task:Pytorch手动实现ResNet18+CIFAR10训练
    @Author:Chen Zhang
    @Date:2024/11/08
"""
import torch
import torch.nn as nn
# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 残差块的定义
class ResidualBlock(nn.Module):
    def __init__(self,inchannels,channels_list,stride=1):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannels,out_channels=channels_list[0],stride=stride,kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(channels_list[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels_list[0],out_channels=channels_list[1],stride=1,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channels_list[1])

        self.shortcut = nn.Sequential()
        if inchannels != channels_list[-1] or stride!=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, channels_list[-1], 1, padding=0, stride=stride,bias=False),
                nn.BatchNorm2d(channels_list[-1])
            )
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)
        out += identity
        return out

# ResNet18层定义
class ResNet18(nn.Module):
    def __init__(self,num_classes=10):
        super(ResNet18,self).__init__()
        # input = 3*224*224
        # (M+2*P-K)/S + 1 == M‘ ==》 P = 2.5 上取整  P = 3
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        # 64*56*56
        self.stage2 = nn.Sequential(
            ResidualBlock(64,[64,64],1),
            ResidualBlock(64,[64,64],1),
        )
        # 64*28*28
        self.stage3 = nn.Sequential(
            ResidualBlock(64,[128,128],2),
            ResidualBlock(128,[128,128],1),
        )
        # 128*14*14
        self.stage4 = nn.Sequential(
            ResidualBlock(128,[256,256],2),
            ResidualBlock(256,[256,256],1),
        )
        # 256*7*7
        self.stage5 = nn.Sequential(
            ResidualBlock(256,[512,512],2),
            ResidualBlock(512,[512,512],1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 全局的平均池化
        self.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.avgpool(out)
        # 此时为1*1*512 应该展平
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out
# 实例化网络
# 实例化网络并迁移到GPU
model = ResNet18(10).to(device)
print(model)
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