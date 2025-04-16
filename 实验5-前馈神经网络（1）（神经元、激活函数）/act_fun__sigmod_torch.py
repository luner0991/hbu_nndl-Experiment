import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 生成输入值
z = torch.linspace(-10, 10, 100)  # 在[-10, 10]范围内生成100个均匀分布的点
# 计算 Logistic 函数（Sigmoid）
logistic_output = F.sigmoid(z)  # 使用 torch.nn.functional.sigmoid 计算 Logistic 函数
# 计算 Tanh 函数
tanh_output = F.tanh(z)  # 使用 torch.tanh 计算 Tanh 函数
# 打印结果
print("输入 z:", z)
print("Logistic 函数输出:", logistic_output)
print("Tanh 函数输出:", tanh_output)
# 可视化结果
plt.figure(figsize=(10, 5))  # 创建一个 10x5 英寸的图形
# 绘制 Logistic 函数
plt.plot(z.numpy(), logistic_output.numpy(), color='purple', label='Logistic Function (Sigmoid)', linestyle='-')  # 绘制 Logistic 函数
# 绘制 Tanh 函数
plt.plot(z.numpy(), tanh_output.numpy(), color='orange', label='Tanh Function', linestyle='--')  # 绘制 Tanh 函数
# 设置图形属性
plt.title('Logistic and Tanh Functions')  # 设置标题
plt.xlabel('z')  # 设置 x 轴标签
plt.ylabel('Function Output')  # 设置 y 轴标签
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # 绘制 y=0 的虚线
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # 绘制 x=0 的虚线
plt.grid()  # 显示网格
plt.legend()  # 显示图例
plt.show()  # 显示图形
