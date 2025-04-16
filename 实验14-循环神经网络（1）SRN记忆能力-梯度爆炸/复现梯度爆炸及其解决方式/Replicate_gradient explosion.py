'''
@Function: 复现梯度爆炸现象
@Author: lxy
@Date: 2024/12/7
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import random
import matplotlib.pyplot as plt
W_list = []
U_list = []
b_list = []
# 计算梯度范数
def custom_print_log(runner):
    model = runner.model
    W_grad_l2, U_grad_l2, b_grad_l2 = 0, 0, 0
    for name, param in model.named_parameters():
        if name == "rnn_model.W":
            W_grad_l2 = torch.norm(param.grad, p=2).numpy()
        if name == "rnn_model.U":
            U_grad_l2 = torch.norm(param.grad, p=2).numpy()
        if name == "rnn_model.b":
            b_grad_l2 = torch.norm(param.grad, p=2).numpy()
    print(f"[Training] W_grad_l2: {W_grad_l2:.5f}, U_grad_l2: {U_grad_l2:.5f}, b_grad_l2: {b_grad_l2:.5f} ")
    W_list.append(W_grad_l2)
    U_list.append(U_grad_l2)
    b_list.append(b_grad_l2)


import sys
sys.path.append("D:\learnning resource-lxy\深度学习-实验\实验14-循环神经网络（1）SRN记忆能力-梯度爆炸\SRN记忆能力")  # 添加SRN记忆能力文件夹路径到sys.path
from SRN记忆能力 import model1
from Runner import Accuracy, RunnerV3,plot_training_loss

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# 训练轮次
num_epochs = 50
# 学习率
lr = 0.2
# 输入数字的类别数
num_digits = 10
# 将数字映射为向量的维度
input_size = 32
# 隐状态向量的维度
hidden_size = 32
# 预测数字的类别数
num_classes = 19
# 批大小
batch_size = 64
# 模型保存目录
save_dir = "./checkpoints"

# 可以设置不同的length进行不同长度数据的预测实验
length = 10
print(f"\n====> Training SRN with data of length {length}.")

# 加载长度为length的数据，
data_path = f"../SRN记忆能力/datasets/{length}"
train_examples, dev_examples, test_examples = model1.load_data(data_path)
train_set, dev_set, test_set = model1.DigitSumDataset(train_examples), model1.DigitSumDataset(dev_examples), model1.DigitSumDataset(
    test_examples)
train_loader = DataLoader(train_set, batch_size=batch_size)
dev_loader = DataLoader(dev_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)
# 实例化模型
base_model = model1.SRN(input_size, hidden_size)
model2 = model1.Model_RNN4SeqClass(base_model, num_digits, input_size, hidden_size, num_classes)
# 指定优化器
optimizer = torch.optim.SGD(lr=lr, params=model2.parameters())
# 定义评价指标
metric = Accuracy()
# 定义损失函数
loss_fn = nn.CrossEntropyLoss(reduction="sum")

# 基于以上组件，实例化Runner
runner = RunnerV3(model2, optimizer, loss_fn, metric)

# 进行模型训练
model_save_path = os.path.join(save_dir, f"srn_explosion_model_{length}.pdparams")
runner.train(train_loader, dev_loader, num_epochs=num_epochs, eval_steps=100, log_steps=1,
             save_path=model_save_path, custom_print_log=custom_print_log)

import matplotlib.pyplot as plt
def plot_grad(W_list, U_list, b_list, save_path, keep_steps=40):
    # 开始绘制图片
    plt.figure()
    # 默认保留前40步的结果
    steps = list(range(keep_steps))
    plt.plot(steps, W_list[:keep_steps], "r-", color="#e4007f", label="W_grad_l2")
    plt.plot(steps, U_list[:keep_steps], "-.", color="#f19ec2", label="U_grad_l2")
    plt.plot(steps, b_list[:keep_steps], "--", color="#000000", label="b_grad_l2")

    plt.xlabel("step")
    plt.ylabel("L2 Norm")
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig(save_path)
    print("image has been saved to: ", save_path)

# 获取保存路径中的目录部分（这里是'./images'）
save_path = os.path.dirname('./images/6.9.pdf')
# 如果目录不存在，则创建它
if not os.path.exists(save_path):
    os.makedirs(save_path)
plot_grad(W_list, U_list, b_list, save_path, keep_steps=100)

# 加载训练过程中效果最好的模型
model_path = os.path.join(save_dir, "srn_explosion_model_20.pdparams")
runner.load_model(model_path)

# 使用测试集评价模型，获取测试集上的预测准确率
score, _ = runner.evaluate(test_loader)
print(f"[SRN] length:{length}, Score: {score: .5f}")


