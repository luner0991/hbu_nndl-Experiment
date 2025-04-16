import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1)
])

data_train = datasets.ImageFolder('train_data', transforms)
data_test = datasets.ImageFolder('test_data', transforms)

train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
test_loader = DataLoader(data_test, batch_size=64, shuffle=True)
for i, data in enumerate(train_loader):
    images, labels = data
    print(images.shape)
    print(labels.shape)
    break

for i, data in enumerate(test_loader):
    images, labels = data
    print(images.shape)
    print(labels.shape)
    break
# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, 3)  # 第一层卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.conv2 = nn.Conv2d(9, 5, 3)  # 第二层卷积层
        self.relu = nn.ReLU()  # 激活函数

        # 全连接层
        self.fc1 = nn.Linear(27 * 27 * 5, 1200)
        self.fc2 = nn.Linear(1200, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        outputs.append(x)  # 保存经过第一层卷积的特征图
        x = self.relu(x)
        outputs.append(x)  # 保存经过 ReLU 激活后的特征图
        x = self.pool(x)
        outputs.append(x)  # 保存经过池化后的特征图
        x = self.conv2(x)
        outputs.append(x)
        x = self.relu(x)
        outputs.append(x)
        x = self.pool(x)
        outputs.append(x)

        x = x.view(-1, 27 * 27 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return outputs

# 加载模型
model = CNN()
model.load_state_dict(torch.load('bestmodel'))
print(model)

# 测试输入数据
x = images[0].unsqueeze(0)
out_put = model(x)

'''
查看训练好的模型特征图
'''
titles = ["Conv1 Output", "ReLU after Conv1", "MaxPool after Conv1",
          "Conv2 Output", "ReLU after Conv2", "MaxPool after Conv2"]  # 每层特征图的标题
for i, feature_map in enumerate(out_put):
    im = np.squeeze(feature_map.detach().numpy())
    im = np.transpose(im, [1, 2, 0])  # 调整通道维度

    plt.figure()
    plt.suptitle(titles[i])  # 设置每个特征图的标题
    num_filters = im.shape[2]  # 特征图的通道数（即滤波器个数）
    for j in range(num_filters):  # 显示每一层的特征图
        ax = plt.subplot(3, 3, j + 1)
        plt.imshow(im[:, :, j], cmap='gray')
        plt.axis('off')  # 关闭坐标轴
    plt.show()

'''
查看训练好的模型的卷积核 
'''
# forward正向传播过程
out_put = model(x)
weights_keys = model.state_dict().keys()
for key in weights_keys:
    print("key :", key)
    # 卷积核通道排列顺序 [kernel_number, kernel_channel, kernel_height, kernel_width]
    if key == "conv1.weight":
        weight_t = model.state_dict()[key].numpy()
        print("weight_t.shape", weight_t.shape)
        k = weight_t[:, 0, :, :]  # 获取第一个卷积核的信息参数
        # show 9 kernel ,1 channel
        plt.figure()

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)  # 参数意义：3：图片绘制行数，5：绘制图片列数，i+1：图的索引
            plt.imshow(k[i, :, :], cmap='gray')
            title_name = 'kernel' + str(i) + ',channel1'
            plt.title(title_name)
        plt.show()

    if key == "conv2.weight":
        weight_t = model.state_dict()[key].numpy()
        print("weight_t.shape", weight_t.shape)
        k = weight_t[:, :, :, :]  # 获取第一个卷积核的信息参数
        print(k.shape)
        print(k)

        plt.figure()
        for c in range(9):
            channel = k[:, c, :, :]
            for i in range(5):
                ax = plt.subplot(2, 3, i + 1)  # 参数意义：3：图片绘制行数，5：绘制图片列数，i+1：图的索引
                plt.imshow(channel[i, :, :], cmap='gray')
                title_name = 'kernel' + str(i) + ',channel' + str(c)
                plt.title(title_name)
            plt.show()
