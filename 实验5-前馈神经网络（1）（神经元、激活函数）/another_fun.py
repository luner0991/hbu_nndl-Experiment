import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 生成输入值
z = torch.linspace(-10, 10, 100)

# 计算激活函数的输出
hard_sigmoid_output = F.hardsigmoid(z)  # Hard-Logistic (Hard-Sigmoid)
hard_tanh_output = F.hardtanh(z)  # Hard-Tanh
elu_output = F.elu(z)  # ELU
softplus_output = F.softplus(z)  # Softplus
swish_output = F.silu(z)  # 使用 SiLU 作为 Swish 的实现

# 可视化结果
plt.figure(figsize=(10, 5))

# 绘制各个激活函数的输出
plt.plot(z.numpy(), hard_sigmoid_output.numpy(), color='purple', label='Hard-Logistic (Hard-Sigmoid)', linestyle='-')
plt.plot(z.numpy(), hard_tanh_output.numpy(), color='blue', label='Hard-Tanh', linestyle='--')
plt.plot(z.numpy(), elu_output.numpy(), color='green', label='ELU', linestyle='-.')
plt.plot(z.numpy(), softplus_output.numpy(), color='red', label='Softplus', linestyle=':')
plt.plot(z.numpy(), swish_output.numpy(), color='orange', label='Swish (SiLU)', linestyle='-')

# 设置图形属性
plt.title('Activation Functions: Hard-Logistic, Hard-Tanh, ELU, Softplus, Swish')
plt.xlabel('z')
plt.ylabel('Function Output')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.legend()
plt.show()
