import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义 ReLU 激活函数
def relu(z):
    return np.maximum(0, z)  # 当 z > 0 返回 z，否则返回 0

# 定义带泄露的 ReLU 激活函数
def leaky_relu(z, negative_slope=0.1):
    a1 = (z > 0) * z
    a2 = (z <= 0) * (negative_slope * z)
    return a1 + a2
# 绘制函数曲线
def plot_activation_functions():
    z = np.linspace(-10, 10, 1000)  # 生成输入值
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制 ReLU 函数
    plt.plot(z, relu(z), color='#e4007f', label="ReLU Function", linestyle='-')

    # 绘制带泄露的 ReLU 函数
    plt.plot(z, leaky_relu(z), color='orange', linestyle='--', label="Leaky ReLU Function (alpha=0.01)")
    # 设置图形属性
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))  # y轴与 x = 0 对齐
    ax.spines['bottom'].set_position(('data', 0))  # x轴与 y = 0 对齐
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # 添加水平基线
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # 添加垂直基线
    plt.title("ReLU and Leaky ReLU Activation Functions")  # 设置标题
    plt.xlabel("Input (z)")  # x轴标签
    plt.ylabel("Output")  # y轴标签
    plt.legend(loc='best', fontsize='large')  # 显示图例
    plt.grid(True)  # 添加网格
    plt.show()  # 显示图形

# 主函数
if __name__ == "__main__":
    plot_activation_functions()
