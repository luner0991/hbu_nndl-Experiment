'''
@Function: 函数x^2的不同优化算法的比较分析及2D可视化
@Author:lxy
@Date:2024/12/19
'''
from op import Op,SimpleBatchGD,Optimizer
import torch
import numpy as np
from matplotlib import pyplot as plt
import copy


# 优化函数
class OptimizedFunction(Op):
    def __init__(self, w):
        super(OptimizedFunction, self).__init__()
        self.w = w
        self.params = {'x': 0}
        self.grads = {'x': 0}

    def forward(self, x):
        self.params['x'] = x
        return torch.matmul(self.w.T, torch.tensor(torch.square(self.params['x']), dtype=torch.float32))

    def backward(self):
        self.grads['x'] = 2 * torch.multiply(self.w.T, self.params['x'])


# 训练函数，记录梯度下降过程中每轮的参数x和损失
def train_f(model, optimizer, x_init, epoch):
    """
    训练函数
    输入：
        - model：被优化函数
        - optimizer：优化器
        - x_init：x初始值
        - epoch：训练回合数
    """
    x = x_init
    all_x = []
    losses = []
    for i in range(epoch):
        all_x.append(copy.copy(x.numpy()))
        loss = model(x)
        losses.append(loss)
        model.backward()
        optimizer.step()
        x = model.params['x']
        print(all_x)
    return torch.tensor(all_x), losses


# 可视化函数，用于绘制更新轨迹
class Visualization(object):
    def __init__(self):
        """
        初始化可视化类
        """
        # 只画出参数x1和x2在区间[-5, 5]的曲线部分
        x1 = np.arange(-5, 5, 0.1)
        x2 = np.arange(-5, 5, 0.1)
        x1, x2 = np.meshgrid(x1, x2)
        self.init_x = torch.tensor([x1, x2])

    def plot_2d(self, model, x, fig_name):
        """
        可视化参数更新轨迹
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        cp = ax.contourf(self.init_x[0], self.init_x[1], model(self.init_x.transpose(0, 1)),
                         colors=['#e4007f', '#f19ec2', '#e86096', '#eb7aaa', '#f6c8dc', '#f5f5f5', '#000000'])
        c = ax.contour(self.init_x[0], self.init_x[1], model(self.init_x.transpose(0, 1)), colors='black')
        cbar = fig.colorbar(cp)
        ax.plot(x[:, 0], x[:, 1], '-o', color='#000000')
        ax.plot(0, 'r*', markersize=18, color='#fefefe')

        ax.set_xlabel('$x1$')
        ax.set_ylabel('$x2$')

        ax.set_xlim((-2, 5))
        ax.set_ylim((-2, 5))
        plt.savefig(fig_name)


# 训练模型并可视化参数更新轨迹
def train_and_plot_f(model, optimizer, epoch, fig_name):
    """
    训练模型并可视化参数更新轨迹
    """
    # 设置x的初始值
    x_init = torch.tensor([3, 4], dtype=torch.float32)
    print('x1 initiate: {}, x2 initiate: {}'.format(x_init[0].numpy(), x_init[1].numpy()))
    x, losses = train_f(model, optimizer, x_init, epoch)
    print(x)
    losses = np.array(losses)

    # 展示x1、x2的更新轨迹
    vis = Visualization()
    vis.plot_2d(model, x, fig_name)

#1.SGD
# 固定随机种子
torch.manual_seed(0)
w = torch.tensor([0.2, 2])
model = OptimizedFunction(w)
opt = SimpleBatchGD(init_lr=0.2, model=model)
train_and_plot_f(model, opt, epoch=40, fig_name='opti-vis-para.pdf')
plt.title("SGD")
plt.show()
#2.AdaGrad
class Adagrad(Optimizer):
    def __init__(self, init_lr, model, epsilon):
        """
        Adagrad 优化器初始化
        输入：
            - init_lr： 初始学习率
            - model：模型，model.params存储模型参数值
            - epsilon：保持数值稳定性而设置的非常小的常数
        """
        super(Adagrad, self).__init__(init_lr=init_lr, model=model)
        self.G = {}
        for key in self.model.params.keys():
            self.G[key] = 0
        self.epsilon = epsilon

    def adagrad(self, x, gradient_x, G, init_lr):
        """
        adagrad算法更新参数，G为参数梯度平方的累计值。
        """
        G += gradient_x ** 2
        x -= init_lr / torch.sqrt(G + self.epsilon) * gradient_x
        return x, G

    def step(self):
        """
        参数更新
        """
        for key in self.model.params.keys():
            self.model.params[key], self.G[key] = self.adagrad(self.model.params[key],
                                                               self.model.grads[key],
                                                               self.G[key],
                                                               self.init_lr)

# 固定随机种子
torch.manual_seed(0)
w = torch.tensor([0.2, 2])
model = OptimizedFunction(w)
opt = Adagrad(init_lr=0.5, model=model, epsilon=1e-7)
train_and_plot_f(model, opt, epoch=50, fig_name='opti-vis-para2.pdf')
plt.title("AdaGrad")
plt.show()
#3.RMSprop
class RMSprop(Optimizer):
    def __init__(self, init_lr, model, beta, epsilon):
        """
        RMSprop优化器初始化
        输入：
            - init_lr：初始学习率
            - model：模型，model.params存储模型参数值
            - beta：衰减率
            - epsilon：保持数值稳定性而设置的常数
        """
        super(RMSprop, self).__init__(init_lr=init_lr, model=model)
        self.G = {}
        for key in self.model.params.keys():
            self.G[key] = 0
        self.beta = beta
        self.epsilon = epsilon

    def rmsprop(self, x, gradient_x, G, init_lr):
        """
        rmsprop算法更新参数，G为迭代梯度平方的加权移动平均
        """
        G = self.beta * G + (1 - self.beta) * gradient_x ** 2
        x -= init_lr / torch.sqrt(G + self.epsilon) * gradient_x
        return x, G

    def step(self):
        """参数更新"""
        for key in self.model.params.keys():
            self.model.params[key], self.G[key] = self.rmsprop(self.model.params[key],
                                                               self.model.grads[key],
                                                               self.G[key],
                                                               self.init_lr)

# 固定随机种子
torch.manual_seed(0)
w = torch.tensor([0.2, 2])
model = OptimizedFunction(w)
opt = RMSprop(init_lr=0.1, model=model, beta=0.9, epsilon=1e-7)
train_and_plot_f(model, opt, epoch=50, fig_name='opti-vis-para3-RMSprop.pdf')
plt.title("RMSprop")
plt.show()
#4.Momentum
class Momentum(Optimizer):
    def __init__(self, init_lr, model, rho):
        """
        Momentum优化器初始化
        输入：
            - init_lr：初始学习率
            - model：模型，model.params存储模型参数值
            - rho：动量因子
        """
        super(Momentum, self).__init__(init_lr=init_lr, model=model)
        self.delta_x = {}
        for key in self.model.params.keys():
            self.delta_x[key] = 0
        self.rho = rho

    def momentum(self, x, gradient_x, delta_x, init_lr):
        """
        momentum算法更新参数，delta_x为梯度的加权移动平均
        """
        delta_x = self.rho * delta_x - init_lr * gradient_x
        x += delta_x
        return x, delta_x

    def step(self):
        """参数更新"""
        for key in self.model.params.keys():
            self.model.params[key], self.delta_x[key] = self.momentum(self.model.params[key],
                                                                      self.model.grads[key],
                                                                      self.delta_x[key],
                                                                      self.init_lr)

# 固定随机种子
torch.manual_seed(0)
w = torch.tensor([0.2, 2])
model = OptimizedFunction(w)
opt = Momentum(init_lr=0.1, model=model, rho=0.9)
train_and_plot_f(model, opt, epoch=50, fig_name='opti-vis-para4-Momentum.pdf')
plt.title("Momentum")
plt.show()
#5.Adam
class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1, beta2, epsilon):
        """
        Adam优化器初始化
        输入：
            - init_lr：初始学习率
            - model：模型，model.params存储模型参数值
            - beta1, beta2：移动平均的衰减率
            - epsilon：保持数值稳定性而设置的常数
        """
        super(Adam, self).__init__(init_lr=init_lr, model=model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.M, self.G = {}, {}
        for key in self.model.params.keys():
            self.M[key] = 0
            self.G[key] = 0
        self.t = 1

    def adam(self, x, gradient_x, G, M, t, init_lr):
        """
        adam算法更新参数
        输入：
            - x：参数
            - G：梯度平方的加权移动平均
            - M：梯度的加权移动平均
            - t：迭代次数
            - init_lr：初始学习率
        """
        M = self.beta1 * M + (1 - self.beta1) * gradient_x
        G = self.beta2 * G + (1 - self.beta2) * gradient_x ** 2
        M_hat = M / (1 - self.beta1 ** t)
        G_hat = G / (1 - self.beta2 ** t)
        t += 1
        x -= init_lr / torch.sqrt(G_hat + self.epsilon) * M_hat
        return x, G, M, t

    def step(self):
        """参数更新"""
        for key in self.model.params.keys():
            self.model.params[key], self.G[key], self.M[key], self.t = self.adam(self.model.params[key],
                                                                                 self.model.grads[key],
                                                                                 self.G[key],
                                                                                 self.M[key],
                                                                                 self.t,
                                                                                 self.init_lr)

# 固定随机种子
torch.manual_seed(0)
w = torch.tensor([0.3, 2])
model = OptimizedFunction(w)
opt = Adam(init_lr=0.2, model=model, beta1=0.9, beta2=0.99, epsilon=1e-7)
train_and_plot_f(model, opt, epoch=50, fig_name='opti-vis-para5-Adam.pdf')
plt.title("Adam ")
plt.show()
