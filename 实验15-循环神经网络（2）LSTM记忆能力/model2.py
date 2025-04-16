'''
@Function: 实现数字求和任务测试LSTM网络的记忆能力，验证LSTM在参数学习时可以有效解决长程依赖问题
@Author: lxy
@Date: 2024/12/7
'''
import os
import random
import torch
import torch.nn as nn
import numpy as np
from Runner import Accuracy,RunnerV3,plot_training_loss
from LSTMselfdefine import LSTM
import sys
from model1 import load_data,DataLoader,Model_RNN4SeqClass,DigitSumDataset
import matplotlib.pyplot as plt

# ===============模型训练===================
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

# 可以设置不同的length进行不同长度数据的预测实验
def train(length):
    print(f"\n====> Training LSTM with data of length {length}.")
    # 固定随机种子
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # 加载长度为length的数据
    data_path = f"../实验14-循环神经网络（1）SRN记忆能力-梯度爆炸/SRN记忆能力/datasets/{length}"
    train_examples, dev_examples, test_examples = load_data(data_path)
    train_set, dev_set, test_set = DigitSumDataset(train_examples), DigitSumDataset(dev_examples), DigitSumDataset(
        test_examples)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    # 实例化模型
    base_model = LSTM(input_size, hidden_size)
    model = Model_RNN4SeqClass(base_model, num_digits, input_size, hidden_size, num_classes)
    # 指定优化器
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    # 定义评价指标
    metric = Accuracy()
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 基于以上组件，实例化Runner
    runner = RunnerV3(model, optimizer, loss_fn, metric)

    # 进行模型训练
    model_save_path = os.path.join(save_dir, f"best_lstm_model_{length}.pdparams")
    runner.train(train_loader, dev_loader, num_epochs=num_epochs, eval_steps=100, log_steps=100,
                 save_path=model_save_path)

    return runner

# ========================多组训练====================
lstm_runners = {}
lengths = [10, 15, 20, 25, 30, 35]
for length in lengths:
    runner = train(length)
    lstm_runners[length] = runner

# 画出训练过程中的损失图
for length in lengths:
    runner = lstm_runners[length]
    save_path = os.path.dirname(f'./images/6.6_{length}.pdf')
    # 如果目录不存在，则创建它
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_training_loss(runner, save_path, sample_step=100)
# =================模型测试=================
lstm_dev_scores = []
lstm_test_scores = []
for length in lengths:
    print(f"Evaluate LSTM with data length {length}.")
    runner = lstm_runners[length]
    # 加载训练过程中效果最好的模型
    model_path = os.path.join(save_dir, f"best_lstm_model_{length}.pdparams")
    runner.load_model(model_path)

    # 加载长度为length的数据
    data_path = f"../实验14-循环神经网络（1）SRN记忆能力-梯度爆炸/SRN记忆能力/datasets/{length}"
    train_examples, dev_examples, test_examples = load_data(data_path)
    test_set = DigitSumDataset(test_examples)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # 使用测试集评价模型，获取测试集上的预测准确率
    score, _ = runner.evaluate(test_loader)
    lstm_test_scores.append(score)
    lstm_dev_scores.append(max(runner.dev_scores))

for length, dev_score, test_score in zip(lengths, lstm_dev_scores, lstm_test_scores):
    print(f"[LSTM] length:{length}, dev_score: {dev_score}, test_score: {test_score: .5f}")

# 在不同长度的验证集和测试集数据上的表现，绘制成图片进行观察=======================
plt.plot(lengths, lstm_dev_scores, '-o', color='#e8609b',  label="LSTM Dev Accuracy")
plt.plot(lengths, lstm_test_scores,'-o', color='#000000', label="LSTM Test Accuracy")

# 绘制坐标轴和图例
plt.ylabel("accuracy", fontsize='large')
plt.xlabel("sequence length", fontsize='large')
plt.legend(loc='lower left', fontsize='x-large')

fig_name = "./images/6.12.pdf"
plt.savefig(fig_name)
plt.show()

