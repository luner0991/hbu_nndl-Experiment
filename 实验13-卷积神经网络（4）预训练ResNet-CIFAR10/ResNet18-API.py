'''
@Function: 使用预训练resnet18(调用API)实现CIFAR-10分类
@Author: lxy
@date: 2024/11/27
'''
import torch
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader,random_split
import numpy as np
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# ==================数据处理================
transforms = transforms.Compose([
    transforms.Resize((32,32)), # 重新设置图片尺寸
    transforms.ToTensor(), # 转换为tensor格式
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])] # 归一化
)

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transforms)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transforms)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 可视化第一张图像
image, label = trainset[0]
print(image.size())
image, label = np.array(image), int(label)
plt.imshow(image.transpose(1, 2, 0))
plt.show()
print(classes[label])


# ======================模型构建=====================
resnet18_model = resnet18(pretrained=True)
#resnet18_model = resnet18(pretrained=False)

# ======================模型训练======================
import torch.nn.functional as F
import torch.optim as opt
from Runner import RunnerV3,Accuracy,plot

# 指定运行设备
torch.cuda.set_device('cuda:0')
# 学习率大小
lr = 0.001
# 批次大小
batch_size = 64
# 加载数据
train_size = int(0.8 * len(trainset))  # 80%的数据作为训练集
dev_size = len(trainset) - train_size  # 剩余的20%作为验证集
test_size =len(testset)
print(f"train_size: {train_size},dev_size :{dev_size},test_szie :{test_size}")
# 随机划分训练集和验证集
train_data, dev_data = random_split(trainset, [train_size, dev_size])
# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size)
dev_loader = DataLoader(dev_data, batch_size=batch_size)
# 定义网络
model = resnet18_model
# 定义优化器，这里使用Adam优化器以及l2正则化策略，相关内容在7.3.3.2和7.6.2中会进行详细介绍
optimizer = opt.Adam(lr=lr, params=model.parameters(), weight_decay=0.005)
# 定义损失函数
loss_fn = F.cross_entropy
# 定义评价指标
metric = Accuracy()
# 实例化RunnerV3
runner = RunnerV3(model, optimizer, loss_fn, metric)
# 启动训练
log_steps = 3000
eval_steps = 3000
runner.train(train_loader, dev_loader, num_epochs=30, log_steps=log_steps,
             eval_steps=eval_steps, save_path="best_model.pdparams")
# 加载最优模型
runner.load_model('best_model.pdparams')
plot(runner, fig_name='cnn-loss4.pdf')

# ======================模型评价=====================
score, loss = runner.evaluate(test_loader)
print("[Test] accuracy/loss: {:.4f}/{:.4f}".format(score, loss))
#=========================模型预测===================

# 获取测试集中的一个batch的数据
for X, label in test_loader:
    logits = runner.predict(X)
    # 多分类，使用softmax计算预测概率
    pred = F.softmax(logits)
    # 获取概率最大的类别
    pred_class = torch.argmax(pred[2]).cpu().numpy()
    label = label[2].data.numpy()
    # 输出真实类别与预测类别
    print("The true category is {} and the predicted category is {}".format(classes[label], classes[pred_class]))
    # 可视化图片
    X = np.array(X)
    X = X[1]
    plt.imshow(X.transpose(1, 2, 0))
    plt.show()
    break