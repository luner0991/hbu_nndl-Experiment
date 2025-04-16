import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
模型结构及参数定义：一个简单的两层神经网络，包含一个隐藏层和一个输出层
    ·输入层：两个输入节点，分别为 x1 和 x2，表示输入的数据
    ·隐藏层：两个神经元，分别为 h1 和 h2。
        输入层与隐藏层之间的权重 :w[0],w[1],w[2],w[3]
        输入：权重 w[0],w[1],w[2],w[3]和输入值 x1, x2 进行加权线性组合得到 in_h1,in_h2
        输出：经过激活函数处理，得到隐藏层输出out_h1 ，out_h2 
    .输出层：两个神经元，分别为 o1 和 o2，用于预测输出
        隐藏层与输出层之间的权重 w[4],w[5],w[6],w[7]
        输入：权重  w[4],w[5],w[6],w[7]和隐藏层输出值out_h1 ，out_h2进行加权线性组合得到 in_o1,in_o2
        输出：经过激活函数处理，得到输出层输出out_o1 ，out_o2 
---------------------------------------------------------------
均方误差（MSE）计算预测值 out_o1, out_o2 与真实值 y1, y2 之间的误差，
根据误差计算每个权重 w[0] 至 w[7] 的梯度，然后通过梯度下降法调整权重。   
---------------------------------------------------------------
'''

# 激活函数 - Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 前向传播过程 - 计算隐藏层和输出层的值
def forward_propagate(x1, x2, y1, y2, w):
    # 计算隐藏层的输入和输出
    in_h1 = w[0] * x1 + w[2] * x2  # 线性组合
    out_h1 = sigmoid(in_h1)  # 激活
    in_h2 = w[1] * x1 + w[3] * x2
    out_h2 = sigmoid(in_h2)

    # 计算输出层的输入和输出
    in_o1 = w[4] * out_h1 + w[6] * out_h2
    out_o1 = sigmoid(in_o1)
    in_o2 = w[5] * out_h1 + w[7] * out_h2
    out_o2 = sigmoid(in_o2)
    # 输出隐藏层和输出层的结果
    print("正向计算，隐藏层h1 ,h2的输出：", end="")
    print(round(out_h1, 5), round(out_h2, 5))  # 四舍五入保留五位小数
    print("正向计算，输出层的最终预测值o1 ,o2：", end="")
    print(round(out_o1, 5), round(out_o2, 5))

    # 计算损失函数（均方误差）
    error = (1 / 2) * ((out_o1 - y1) ** 2 + (out_o2 - y2) ** 2)
    print("误差为：",error)
    return out_o1, out_o2, out_h1, out_h2, error  # 返回结果及损失


# 反向传播函数 - 计算损失函数相对于每个权重的梯度
def back_propagate(out_o1, out_o2, out_h1, out_h2, y1, y2, w):
    # 输出层的误差
    d_o1 = out_o1 - y1
    d_o2 = out_o2 - y2
    # 计算输出层到隐藏层的权重梯度
    d_w = np.zeros(8)
    d_w[4] = d_o1 * out_o1 * (1 - out_o1) * out_h1
    d_w[6] = d_o1 * out_o1 * (1 - out_o1) * out_h2
    d_w[5] = d_o2 * out_o2 * (1 - out_o2) * out_h1
    d_w[7] = d_o2 * out_o2 * (1 - out_o2) * out_h2
    # 计算隐藏层到输入层的权重梯度
    d_h1 = (d_o1 * w[4] + d_o2 * w[5]) * out_h1 * (1 - out_h1)
    d_h2 = (d_o1 * w[6] + d_o2 * w[7]) * out_h2 * (1 - out_h2)
    d_w[0] = d_h1 * x1
    d_w[2] = d_h1 * x2
    d_w[1] = d_h2 * x1
    d_w[3] = d_h2 * x2
    print('更新后的权值：',d_w)
    return d_w

# 更新权值
def update_weight(w, d_w, step):
    return w - step * d_w

if __name__ == "__main__":
    # 定义初始权值参数 w
    w = np.array([0.2, -0.4, 0.5, 0.6, 0.1, -0.5, -0.3, 0.8])
    x1, x2 = 0.5, 0.3  # 输入
    y1, y2 = 0.2, 0.7  # 真实输出
    print("输入值 x1, x2:", x1, x2)
    print("目标输出值 y1, y2:", y1, y2)
    print("初始权值:", w.round(2))
    Error = []
    epoh = 200  # 训练次数
    step = float(input('请输入步长：'))

    for i in range(epoh):
        print("=====第{}轮=====".format(i + 1))
        out_o1, out_o2, out_h1, out_h2, error = forward_propagate(x1, x2, y1, y2, w)
        Error.append(error)
        d_w = back_propagate(out_o1, out_o2, out_h1, out_h2, y1, y2, w)
        w = update_weight(w, d_w, step)

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(range(epoh), Error)
    plt.xlabel('迭代轮次')
    plt.ylabel('均方误差')
    plt.title('步长为：{}'.format(step))
    plt.show()
