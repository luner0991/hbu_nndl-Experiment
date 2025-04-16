import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def forward(x, w, y):
    h1 = sigmoid(w[0] * x[0] + w[2] * x[1])
    h2 = sigmoid(w[1] * x[0] + w[3] * x[1])
    o1 = sigmoid(w[4] * h1 + w[6] * h2)
    o2 = sigmoid(w[5] * h1 + w[7] * h2)
    error = ((o1 - y[0]) ** 2 + (o2 - y[1]) ** 2) / 2
    return o1, o2, error


def loss(x, y, w):
    y_pre, _, error = forward(x, w, y)
    return error


def train(x, y, w, epoch, step):
    errors = []
    for i in range(epoch):
        print("\n=====第" + str(i + 1) + "轮=====")
        w.grad = None  # 清零梯度
        error = loss(x, y, w)  # 计算损失
        errors.append(error.item())

        error.backward()  # 反向传播
        with torch.no_grad():
            w -= step * w.grad  # 更新权重
        print(f'均方误差={error.item():.5f},\n 更新后的权值={w.tolist()}')
        print("正向计算，预测值:", round(forward(x, w, y)[0].item(), 5), round(forward(x, w, y)[1].item(), 5))
        print("w的梯度:", [round(grad, 5) for grad in w.grad])

    return errors

if __name__ == "__main__":
    x = torch.tensor([0.5, 0.3])  # 输入
    y = torch.tensor([0.2, 0.7])  # 真实标签
    print("输入值",x)
    print("真实输出值",y)
    w = torch.zeros(8, requires_grad=True)  # 权重初始值
  # 权重初始值

    epoch =200  # 训练轮次
    step = 1 # 步长
    errors = train(x, y, w, epoch, step)

    # 画图
    plt.plot(range(epoch), errors)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.xlabel('迭代轮次')
    plt.ylabel('均方误差')
    plt.title(f'步长为：{step}')
    plt.show()
