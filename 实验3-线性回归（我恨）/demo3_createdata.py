import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统下可以使用SimHei字体
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False
'''
输入参数：
    func: 线性函数（可调用）
    interval: 自变量x的取值范围（元组）
    num: 样本数量（int类型）
    noise: 噪声标准差（float类型，控制数据的随机扰动）
    add_outlier: 是否添加异常点（布尔值）
    outlier_ratio: 异常点的比例（float类型）
输出：
    x: 生成的自变量（tensor）
    y: 生成的因变量（tensor）
目标：
    创建一个关于线性模型的数据集，并生成噪声和异常点
'''
def linear_func(x, w=1.2, b=0.5):
    y = w * x + b
    return y
def create_data(func, interval, num, noise=2, add_outlier=False, outlier_ratio=0.05):
    # 均匀采样自变量x
    x = torch.rand(num, 1) * (interval[1] - interval[0]) + interval[0]  # 在指定区间内均匀采样
    y = func(x)

    # 生成高斯分布的噪声并添加到y中
    epsilon = torch.normal(0, noise, y.shape)
    y = y + epsilon

    # 如果需要添加异常点
    if add_outlier:
        outlier_num = int(len(y) * outlier_ratio)  # 计算异常点数量
        if outlier_num > 0:
            outlier_idx = torch.randint(0, len(y), (outlier_num,))  # 随机选择异常点索引
            y[outlier_idx] = y[outlier_idx] * 5  # 将异常点的值放大5倍

    return x, y


# 创建数据集，指定线性函数、取值范围、样本数量、噪声标准差、异常点占比
data_x, data_y = create_data(linear_func, (-10, 10), 150, noise=2, add_outlier=True, outlier_ratio=0.05)
# 将数据集分割为训练集和测试集
data_x_train, data_y_train = data_x[0:100], data_y[0:100]  # 训练集前100个样本
data_x_test, data_y_test = data_x[100:150], data_y[100:150]  # 测试集后50个样本
# 可视化生成的数据
plt.figure(1)
# 用红色点绘制训练集，用绿色点绘制测试集
plt.plot(data_x_train.numpy(), data_y_train.numpy(), '.r', label="训练集")
plt.plot(data_x_test.numpy(), data_y_test.numpy(), '.g', label="测试集")
plt.legend()  # 添加图例
plt.title("线性模型生成的数据集（包含噪声和异常点）")  # 添加标题
plt.show()
