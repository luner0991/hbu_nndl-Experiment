import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 生成输入值
z = torch.linspace(-10, 10, 100)  # 在[-10, 10]范围内生成100个均匀分布的点
# 计算 ReLU 函数
relu_output=F.relu(z)
# 计算带泄露的 ReLU 函数
leaky_relu_output = F.leaky_relu(z, negative_slope=0.1)
# 打印结果
print("输入 z:", z)
print("ReLU 函数输出:", relu_output)
print("带泄露的 ReLU 函数输出:", leaky_relu_output)

# 可视化结果
plt.figure(figsize=(10, 5))  # 创建一个 10x5 英寸的图形
# 绘制 ReLU 函数
plt.plot(z.numpy(), relu_output.numpy(), color='purple', label='ReLU Function', linestyle='-')  # 绘制 ReLU 函数
# 绘制带泄露的 ReLU 函数
plt.plot(z.numpy(), leaky_relu_output.numpy(), color='orange', label='Leaky ReLU Function', linestyle='--')  # 绘制 Leaky ReLU 函数
# 设置图形属性
plt.title('ReLU and Leaky ReLU Functions')  # 设置标题
plt.xlabel('z')  # 设置 x 轴标签
plt.ylabel('Function Output')  # 设置 y 轴标签
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # 绘制 y=0 的虚线
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # 绘制 x=0 的虚线
plt.grid()  # 显示网格
plt.legend()  # 显示图例
plt.show()  # 显示图形
