import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统下可以使用SimHei字体
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

def linear_func(x, w=1.2, b=0.5):
    y = w * x + b
    return y

def create_data(func, interval, num, noise=2, add_outlier=False, outlier_ratio=0.05):
    # 均匀采样自变量x
    x = torch.rand(num, 1) * (interval[1] - interval[0]) + interval[0]
    y = func(x)

    # 生成高斯分布的噪声并添加到y中
    epsilon = torch.normal(0, noise, y.shape)
    y = y + epsilon

    # 如果需要添加异常点
    if add_outlier:
        outlier_num = int(len(y) * outlier_ratio)
        if outlier_num > 0:
            outlier_idx = torch.randint(0, len(y), (outlier_num,))
            y[outlier_idx] = y[outlier_idx] * 5

    return x, y

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        # 初始化模型参数
        self.params = {}
        self.params['w'] = nn.Parameter(torch.randn(size=(input_size, 1)))  # 权重初始化为随机值
        self.params['b'] = nn.Parameter(torch.zeros([1]))  # 偏置初始化为零

    def forward(self, X, y_true=None):
        N, D = X.shape
        assert D == self.params['w'].shape[0], "输入维度与模型参数不匹配"
        y_pred = torch.matmul(X, self.params['w']) + self.params['b']

        mse_loss = None
        if y_true is not None:
            error = y_pred - y_true
            mse_loss = torch.mean(error ** 2)  # 计算均方损失
        return y_pred, mse_loss  # 返回预测值和损失（如果有）

    def optimizer(self, x, y):
        x_mean = torch.mean(x, dim=0, keepdim=True)  # 特征均值
        tmp = x - x_mean
        y_mean = torch.mean(y)

        # 计算权重
        w = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(tmp.T, tmp)), tmp.T),
            (y - y_mean)
        )
        b = y_mean - torch.matmul(x_mean, w)

        # 更新模型参数
        self.params['w'].data = w
        self.params['b'].data = b

# 创建数据集
data_x, data_y = create_data(linear_func, (-10, 10), 150, noise=2, add_outlier=True, outlier_ratio=0.05)
data_x_train, data_y_train = data_x[0:100], data_y[0:100]  # 训练集
data_x_test, data_y_test = data_x[100:150], data_y[100:150]  # 测试集

# 实例化线性模型
model = LinearModel(input_size=1)

# 使用最优解析解更新模型参数
model.optimizer(data_x_train, data_y_train)

# 在训练集上进行前向传播计算并计算损失
y_pred_train, mse_loss_train = model(data_x_train, data_y_train)

# 打印输出
print("训练集损失:", mse_loss_train.item())  # 打印训练集损失

# 输出训练参数与真实参数的对比
true_w = 1.2  # 真实的权重
true_b = 0.5  # 真实的偏置
learned_w = model.params['w'].item()  # 学习到的权重
learned_b = model.params['b'].item()  # 学习到的偏置

print(f"真实权重: {true_w}, 学习到的权重: {learned_w}")
print(f"真实偏置: {true_b}, 学习到的偏置: {learned_b}")

# 在测试集上进行前向传播计算并计算损失
y_pred_test, mse_loss_test = model(data_x_test, data_y_test)

# 打印输出测试集损失
print("测试集损失:", mse_loss_test.item())

# 输出测试参数与真实参数的对比
learned_w_test = model.params['w'].item()  # 学习到的权重
learned_b_test = model.params['b'].item()  # 学习到的偏置

print(f"真实权重: {true_w}, 学习到的权重: {learned_w_test}")
print(f"真实偏置: {true_b}, 学习到的偏置: {learned_b_test}")

# 可视化生成的数据
plt.figure(1)
plt.plot(data_x_train.numpy(), data_y_train.numpy(), '.r', label="训练集")
plt.plot(data_x_test.numpy(), data_y_test.numpy(), '.g', label="测试集")
plt.plot(data_x_train.numpy(), y_pred_train.detach().numpy(), color='blue', label='拟合线')  # 绘制拟合线
plt.legend()  # 添加图例
plt.title("线性模型生成的数据集（包含噪声和异常点）")  # 添加标题
plt.show()