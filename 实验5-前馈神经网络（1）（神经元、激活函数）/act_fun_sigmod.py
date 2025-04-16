import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义激活函数
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))
def tanh(z):
    exp_z = np.exp(z)  # 计算 e^z
    exp_neg_z = np.exp(-z)  # 计算 e^{-z}
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)  # 返回 Tanh 结果

# 绘制函数曲线
def plot_activation_functions():
    z = np.linspace(-10, 10, 10000)  # 生成输入值
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(z, logistic(z), color='#e4007f', label="Logistic Function")
    plt.plot(z, tanh(z), color='#f19ec2', linestyle='--', label="Tanh Function")
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    plt.axhline(0, color='grey', lw=0.5, linestyle='--')  # 添加水平基线
    plt.axvline(0, color='grey', lw=0.5, linestyle='--')  # 添加垂直基线
    plt.title("Activation Functions")  # 添加标题
    plt.xlabel("Input (z)")  # x轴标签
    plt.ylabel("Output")  # y轴标签
    plt.legend(loc='lower right', fontsize='large')
    plt.grid(True)  # 添加网格
    plt.show()  # 显示图形
# 主函数
if __name__ == "__main__":
    plot_activation_functions()
