import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

# 定义残差单元 (ResBlock)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        super(ResBlock, self).__init__()
        self.use_residual = use_residual

        # 第一个卷积层，卷积核大小为3×3
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积层，卷积核大小为3×3，不改变输入特征图的形状，步长为1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 如果输入和输出形状不一致，则需要使用1×1卷积调整输入形状
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.use = True
        else:
            self.use = False

        if self.use:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            if self.use:  # 如果为真，对x进行1×1卷积，将形状调整为与conv2的输出一致
                identity = self.shortcut(identity)

        out += identity  # 加上残差
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes=1000, use_residual=True):
        super(ResNet18, self).__init__()

        # 第一个卷积层和池化层
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第二个stage，包含两个残差单元
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2, use_residual=True),
            ResBlock(128, 128, stride=1, use_residual=True),
        )

        # 第三个stage，包含两个残差单元
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2, use_residual=True),
            ResBlock(256, 256, stride=1, use_residual=True),
        )

        # 第四个stage，包含两个残差单元
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2, use_residual=True),
            ResBlock(512, 512, stride=1, use_residual=True),
        )

        # 全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局的平均池化
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # 展平为一维
        out = self.fc(out)  # 全连接层
        return out



# 加载HAPI中的resnet18预训练模型
hapi_model = models.resnet18(pretrained=True)
print(hapi_model)

# 自定义ResNet18模型，支持输入通道数和输出类别数的自定义
model = ResNet18(in_channels=3, num_classes=1000, use_residual=True)

# 提取HAPI模型的权重参数
hapi_weights = hapi_model.state_dict()

# 过滤掉不需要的权重（例如，fc层的权重与自定义模型不匹配）
model_weights = model.state_dict()

# 将官方模型的权重映射到自定义模型的相应部分
# 由于fc层的输出类别数不同，忽略最后一层的权重
for name, param in hapi_weights.items():
    if name in model_weights and name != 'fc.weight' and name != 'fc.bias':
        model_weights[name] = param

# 将权重加载到自定义模型中
model.load_state_dict(model_weights)

# 检查加载后的模型是否正确
model.eval()
hapi_model.eval()

# 生成一个随机输入数据作为测试，形状为[2, 3, 32, 32]，模拟一张32x32的RGB图像
inputs = np.random.randn(2, 3, 32, 32).astype('float32')  # 生成float32类型的输入数据
x = torch.tensor(inputs)  # 转换为PyTorch张量，数据类型已经是float32

# 通过自定义模型和HAPI模型进行推理
output = model(x)  # 自定义模型的输出
hapi_out = hapi_model(x)  # 官方ResNet18模型的输出

# 计算两个模型输出的差异
diff = output - hapi_out

# 取差异最大的值
max_diff = torch.max(diff)

# 打印最大差异
print("Max difference between outputs: ", max_diff.item())
