import matplotlib.pyplot as plt
# 初始化两个空列表，分别存储训练损失和验证损失
train_losses = []
dev_losses = []

# 从loss.txt文件读取数据
with open('losses.txt', 'r') as f:
    lines = f.readlines()

    # 读取训练损失部分
    train_start = lines.index('Train Losses:\n') + 1
    dev_start = lines.index('Dev Losses:\n') + 1

    # 读取训练损失
    for line in lines[train_start:dev_start - 1]:
        line = line.strip()  # 去掉前后空格
        if line:  # 如果该行不为空
            try:
                train_losses.append(float(line))  # 尝试将该行转换为浮动数值
            except ValueError:
                print(f"跳过无效数据: {line}")  # 如果无法转换为数字，打印跳过的无效数据

    # 读取验证损失
    for line in lines[dev_start:]:
        line = line.strip()  # 去掉前后空格
        if line:  # 如果该行不为空
            try:
                dev_losses.append(float(line))  # 尝试将该行转换为浮动数值
            except ValueError:
                print(f"跳过无效数据: {line}")  # 如果无法转换为数字，打印跳过的无效数据

# 绘制训练损失和验证损失的图形
epochs = range(1, len(train_losses) + 1)  # 生成每个epoch的数字

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, dev_losses, label='Dev Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Dev Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
