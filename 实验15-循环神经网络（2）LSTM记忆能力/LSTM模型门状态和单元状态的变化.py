'''
@Function: 观察当LSTM在处理一条数字序列的时候，相应门和单元状态
@Author: lxy
@Date: 2024/12/7
'''
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
import os
import random
import torch
import torch.nn as nn
import numpy as np
from Runner import Accuracy,RunnerV3,plot_training_loss
import sys
from model1 import load_data,DataLoader,Model_RNN4SeqClass,DigitSumDataset
import matplotlib.pyplot as plt

# 更新LSTM和相关参数，定义相应列表进行存储这些门和单元状态在每个时刻的向量
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, para_attr=xavier_uniform):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 初始化模型参数
        self.W_i = torch.nn.Parameter(para_attr(torch.empty(size=[input_size, hidden_size], dtype=torch.float32)))
        self.W_f = torch.nn.Parameter(para_attr(torch.empty(size=[input_size, hidden_size], dtype=torch.float32)))
        self.W_o = torch.nn.Parameter(para_attr(torch.empty(size=[input_size, hidden_size], dtype=torch.float32)))
        self.W_c = torch.nn.Parameter(para_attr(torch.empty(size=[input_size, hidden_size], dtype=torch.float32)))
        self.U_i = torch.nn.Parameter(para_attr(torch.empty(size=[hidden_size, hidden_size], dtype=torch.float32)))
        self.U_f = torch.nn.Parameter(para_attr(torch.empty(size=[hidden_size, hidden_size], dtype=torch.float32)))
        self.U_o = torch.nn.Parameter(para_attr(torch.empty(size=[hidden_size, hidden_size], dtype=torch.float32)))
        self.U_c = torch.nn.Parameter(para_attr(torch.empty(size=[hidden_size, hidden_size], dtype=torch.float32)))
        self.b_i = torch.nn.Parameter(para_attr(torch.empty(size=[1, hidden_size], dtype=torch.float32)))
        self.b_f = torch.nn.Parameter(para_attr(torch.empty(size=[1, hidden_size], dtype=torch.float32)))
        self.b_o = torch.nn.Parameter(para_attr(torch.empty(size=[1, hidden_size], dtype=torch.float32)))
        self.b_c = torch.nn.Parameter(para_attr(torch.empty(size=[1, hidden_size], dtype=torch.float32)))

    # 初始化状态向量和隐状态向量
    def init_state(self, batch_size):
        hidden_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)
        cell_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)
        return hidden_state, cell_state

    # 定义前向计算
    def forward(self, inputs, states=None):
        batch_size, seq_len, input_size = inputs.shape  # inputs batch_size x seq_len x input_size

        if states is None:
            states = self.init_state(batch_size)
        hidden_state, cell_state = states

        # 定义相应的门状态和单元状态向量列表
        self.Is = []
        self.Fs = []
        self.Os = []
        self.Cs = []
        # 初始化状态向量和隐状态向量
        cell_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)
        hidden_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)

        # 执行LSTM计算，包括：隐藏门、输入门、遗忘门、候选状态向量、状态向量和隐状态向量
        for step in range(seq_len):
            input_step = inputs[:, step, :]
            I_gate = F.sigmoid(torch.matmul(input_step, self.W_i) + torch.matmul(hidden_state, self.U_i) + self.b_i)
            F_gate = F.sigmoid(torch.matmul(input_step, self.W_f) + torch.matmul(hidden_state, self.U_f) + self.b_f)
            O_gate = F.sigmoid(torch.matmul(input_step, self.W_o) + torch.matmul(hidden_state, self.U_o) + self.b_o)
            C_tilde = F.tanh(torch.matmul(input_step, self.W_c) + torch.matmul(hidden_state, self.U_c) + self.b_c)
            cell_state = F_gate * cell_state + I_gate * C_tilde
            hidden_state = O_gate * F.tanh(cell_state)
            # 存储门状态向量和单元状态向量
            self.Is.append(I_gate.numpy().copy())
            self.Fs.append(F_gate.numpy().copy())
            self.Os.append(O_gate.numpy().copy())
            self.Cs.append(cell_state.numpy().copy())
        return hidden_state
# 训练轮次
num_epochs = 500
# 学习率
lr = 0.001
# 输入数字的类别数
num_digits = 10
# 将数字映射为向量的维度
input_size = 32
# 隐状态向量的维度
hidden_size = 32
# 预测数字的类别数
num_classes = 19
# 批大小
batch_size = 8
# 模型保存目录
save_dir = "./checkpoints"

# =======================实例化模型===========================
base_model = LSTM(input_size, hidden_size)
model = Model_RNN4SeqClass(base_model, num_digits, input_size, hidden_size, num_classes)
# 指定优化器
optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
# 定义评价指标
metric = Accuracy()
# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 基于以上组件，重新实例化Runner
runner = RunnerV3(model, optimizer, loss_fn, metric)

length = 10
# 加载训练过程中效果最好的模型
model_path = os.path.join(save_dir, f"best_lstm_model_{length}.pdparams")
runner.load_model(model_path)

import seaborn as sns
def plot_tensor(inputs, tensor, save_path, vmin=0, vmax=1):
    tensor = np.stack(tensor, axis=0)
    tensor = np.squeeze(tensor, 1).T

    plt.figure(figsize=(16, 6))
    # vmin, vmax定义了色彩图的上下界
    ax = sns.heatmap(tensor, vmin=vmin, vmax=vmax)
    ax.set_xticklabels(inputs)
    ax.figure.savefig(save_path)
    plt.show()
# 定义模型输入
inputs = [6, 7, 0, 0, 1, 0, 0, 0, 0, 0]
X = torch.tensor(inputs.copy())
X = X.unsqueeze(0)
# 进行模型预测，并获取相应的预测结果
logits = runner.predict(X)
predict_label = torch.argmax(logits, dim=-1)
print(f"predict result: {predict_label.numpy()[0]}")

# 输入门
Is = runner.model.rnn_model.Is
plot_tensor(inputs, Is, save_path="./images/6.13_I.pdf")
# 遗忘门
Fs = runner.model.rnn_model.Fs
plot_tensor(inputs, Fs, save_path="./images/6.13_F.pdf")
# 输出门
Os = runner.model.rnn_model.Os
plot_tensor(inputs, Os, save_path="./images/6.13_O.pdf")
# 单元状态
Cs = runner.model.rnn_model.Cs
plot_tensor(inputs, Cs, save_path="./images/6.13_C.pdf", vmin=-5, vmax=5)