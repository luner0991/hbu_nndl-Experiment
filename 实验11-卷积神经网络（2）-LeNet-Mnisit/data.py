import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 加载数据集
train_set, dev_set, test_set = json.load(gzip.open(r'D:\i don`t want to open\深度学习-实验\实验11-卷积神经网络（2）-LeNet-Mnisit\mnist.json.gz', 'rb'))

# 获取前3000个训练样本，200个验证样本和200个测试样本
train_images, train_labels = train_set[0][:3000], train_set[1][:3000]
dev_images, dev_labels = dev_set[0][:200], dev_set[1][:200]
test_images, test_labels = test_set[0][:200], test_set[1][:200]
train_set, dev_set, test_set = [train_images, train_labels], [dev_images, dev_labels], [test_images, test_labels]
 # 打印数据集长度
print('Length of train/dev/test set: {}/{}/{}'.format(len(train_images), len(dev_images), len(test_images)))


image, label = train_set[0][0], train_set[1][0]
image, label = np.array(image).astype('float32'), int(label)
# 原始图像数据为长度784的行向量，需要调整为[28,28]大小的图像
image = np.reshape(image, [28, 28])
# 打印图像的像素值
print("图像的像素值为：")
print(image)
image = Image.fromarray(image.astype('uint8'), mode='L')

print("The number in the picture is {}".format(label))
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.show()

# 数据预处理
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# 数据预处理 将图像的尺寸修改为32*32，转换为tensor形式。并且将输入图像分布改为均值为0，标准差为1的正态分布
transforms = transforms.Compose(
    [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.], std=[1.0])])

# 数据集类的定义 定义MNIST_dataset类，继承dataset类
class MNIST_dataset(Dataset):
    # 初始化数据集，接收一个数据集dataset,转换操作transform  测试操作mode='train'
    def __init__(self, dataset, transforms, mode='train'):
        self.mode = mode
        self.transforms = transforms
        self.dataset = dataset

    # 根据索引idx从数据集中获取样本。
    def __getitem__(self, idx):
        # 获取图像和标签
        image, label = self.dataset[0][idx], self.dataset[1][idx]
        # 将图像转换为float32类型
        image, label = np.array(image).astype('float32'), int(label)
        image = np.reshape(image, [28, 28])  # 重塑形状

        # 将重塑后的图像转换为Image对象，应用转换操作
        image = Image.fromarray(image.astype('uint8'), mode='L')
        image = self.transforms(image)

        return image, label

    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.dataset[0])

# 加载 mnist 数据集 这些数据集在MNIST_dataset类中被初始化，并用于训练、测试和开发模型
train_dataset = MNIST_dataset(dataset=train_set, transforms=transforms, mode='train')
test_dataset = MNIST_dataset(dataset=test_set, transforms=transforms, mode='test')
dev_dataset = MNIST_dataset(dataset=dev_set, transforms=transforms, mode='dev')
