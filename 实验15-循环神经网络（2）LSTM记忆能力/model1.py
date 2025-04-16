'''
@Function: 实现数字求和任务测试简单循环网络的记忆能力，验证简单循环网络在参数学习时存在长程依赖问题
@Author: lxy
@Date: 2024/12/5
'''
import random
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
import torch.nn.functional as F
import time
from Runner import Accuracy, RunnerV3,plot_training_loss
import matplotlib.pyplot as plt

# ======================数据划分=========================
def load_data(data_path):
    # 加载训练集
    train_examples = []
    train_path = os.path.join(data_path, "train.txt")
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # 解析一行数据，将其处理为数字序列seq和标签label
            items = line.strip().split("\t")
            seq = [int(i) for i in items[0].split(" ")]
            label = int(items[1])
            train_examples.append((seq, label))

    # 加载验证集
    dev_examples = []
    dev_path = os.path.join(data_path, "dev.txt")
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # 解析一行数据，将其处理为数字序列seq和标签label
            items = line.strip().split("\t")
            seq = [int(i) for i in items[0].split(" ")]
            label = int(items[1])
            dev_examples.append((seq, label))

    # 加载测试集
    test_examples = []
    test_path = os.path.join(data_path, "test.txt")
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # 解析一行数据，将其处理为数字序列seq和标签label
            items = line.strip().split("\t")
            seq = [int(i) for i in items[0].split(" ")]
            label = int(items[1])
            test_examples.append((seq, label))

    return train_examples, dev_examples, test_examples

# # 设定加载的数据集的长度
# length = 5
# # 该长度的数据集的存放目录
# data_path = f"./datasets/{length}"
# # 加载该数据集
# train_examples, dev_examples, test_examples = load_data(data_path)
# print("dev example:", dev_examples[:2])
# print("训练集数量：", len(train_examples))
# print("验证集数量：", len(dev_examples))
# print("测试集数量：", len(test_examples))

class DigitSumDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        example = self.data[idx]
        seq = torch.tensor(example[0], dtype=torch.int64)
        label = torch.tensor(example[1], dtype=torch.int64)
        return seq, label

    def __len__(self):
        return len(self.data)

#=========================================模型构建========================================
'''
嵌入层
'''
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 para_attr=xavier_uniform):
        super(Embedding, self).__init__()
        # 定义嵌入矩阵
        W = torch.zeros(size=[num_embeddings, embedding_dim], dtype=torch.float32)
        self.W = torch.nn.Parameter(W)
        xavier_uniform(W)

    def forward(self, inputs):
        # 根据索引获取对应词向量
        embs = self.W[inputs]
        return embs

'''
SRN层
'''
torch.manual_seed(0)
class SRN(nn.Module):
    def __init__(self, input_size, hidden_size, W_attr=None, U_attr=None, b_attr=None):
        super(SRN, self).__init__()
        # 嵌入向量的维度
        self.input_size = input_size
        # 隐状态的维度
        self.hidden_size = hidden_size
        # 定义模型参数W，其shape为 input_size x hidden_size
        if W_attr == None:
            W = torch.zeros(size=[input_size, hidden_size], dtype=torch.float32)
        else:
            W = torch.tensor(W_attr, dtype=torch.float32)
        self.W = torch.nn.Parameter(W)
        # 定义模型参数U，其shape为hidden_size x hidden_size
        if U_attr == None:
            U = torch.zeros(size=[hidden_size, hidden_size], dtype=torch.float32)
        else:
            U = torch.tensor(U_attr, dtype=torch.float32)
        self.U = torch.nn.Parameter(U)
        # 定义模型参数b，其shape为 1 x hidden_size
        if b_attr == None:
            b = torch.zeros(size=[1, hidden_size], dtype=torch.float32)
        else:
            b = torch.tensor(b_attr, dtype=torch.float32)
        self.b = torch.nn.Parameter(b)

    # 初始化向量
    def init_state(self, batch_size):
        hidden_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)
        return hidden_state

    # 定义前向计算
    def forward(self, inputs, hidden_state=None):
        # inputs: 输入数据, 其shape为batch_size x seq_len x input_size
        batch_size, seq_len, input_size = inputs.shape

        # 初始化起始状态的隐向量, 其shape为 batch_size x hidden_size
        if hidden_state is None:
            hidden_state = self.init_state(batch_size)

        # 循环执行RNN计算
        for step in range(seq_len):
            # 获取当前时刻的输入数据step_input, 其shape为 batch_size x input_size
            step_input = inputs[:, step, :]
            # 获取当前时刻的隐状态向量hidden_state, 其shape为 batch_size x hidden_size
            hidden_state = F.tanh(torch.matmul(step_input, self.W) + torch.matmul(hidden_state, self.U) + self.b)
        return hidden_state

'''
线性层直接nn.Linear
'''
'''
汇总
'''
# 基于RNN实现数字预测的模型
class Model_RNN4SeqClass(nn.Module):
    def __init__(self, model, num_digits, input_size, hidden_size, num_classes):
        super(Model_RNN4SeqClass, self).__init__()
        # 传入实例化的RNN层，例如SRN
        self.rnn_model = model
        # 词典大小
        self.num_digits = num_digits
        # 嵌入向量的维度
        self.input_size = input_size
        # 定义Embedding层
        self.embedding = Embedding(num_digits, input_size)
        # 定义线性层
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        # 将数字序列映射为相应向量
        inputs_emb = self.embedding(inputs)
        # 调用RNN模型
        hidden_state = self.rnn_model(inputs_emb)
        # 使用最后一个时刻的状态进行数字预测
        logits = self.linear(hidden_state)
        return logits

# =========================模型训练=============================
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


# 通过指定length进行不同长度数据的实验
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
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 通过指定length进行不同长度数据的实验
def train(length):
    print(f"\n====> Training SRN with data of length {length}.")
    # 固定随机种子
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # 加载长度为length的数据
    data_path = f"./datasets/{length}"
    train_examples, dev_examples, test_examples = load_data(data_path)
    train_set, dev_set, test_set = DigitSumDataset(train_examples), DigitSumDataset(dev_examples), DigitSumDataset(
        test_examples)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    # 实例化模型
    base_model = SRN(input_size, hidden_size)
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
    model_save_path = os.path.join(save_dir, f"best_srn_model_{length}.pdparams")
    runner.train(train_loader, dev_loader, num_epochs=num_epochs, eval_steps=100, log_steps=100,
                 save_path=model_save_path)

    return runner


# 判断是否为主模块运行入口，若是则执行后续代码，若被导入则不执行
if __name__ == "__main__":
    srn_runners = {}
    lengths = [10, 15, 20, 25, 30, 35]
    for length in lengths:
        runner = train(length)
        srn_runners[length] = runner
    # 画出训练过程中的损失图
    for length in lengths:
        runner = srn_runners[length]
        #fig_name = f"./images/6.6_{length}.pdf"
        # 如果目录不存在，则创建它
        #if not os.path.exists(fig_name):
        #    os.makedirs(fig_name)

        # 获取保存路径中的目录部分（这里是'./images'）
        save_path = os.path.dirname(f'./images/6.6_{length}.pdf')
        # 如果目录不存在，则创建它
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot_training_loss(runner, save_path, sample_step=100)

    # ==================模型评价=======================================
    srn_dev_scores = []
    srn_test_scores = []
    for length in lengths:
        print(f"Evaluate SRN with data length {length}.")
        runner = srn_runners[length]
        # 加载训练过程中效果最好的模型
        model_path = os.path.join(save_dir, f"best_srn_model_{length}.pdparams")
        runner.load_model(model_path)

        # 加载长度为length的数据
        data_path = f"./datasets/{length}"
        train_examples, dev_examples, test_examples = load_data(data_path)
        test_set = DigitSumDataset(test_examples)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        # 使用测试集评价模型，获取测试集上的预测准确率
        score, _ = runner.evaluate(test_loader)
        srn_test_scores.append(score)
        srn_dev_scores.append(max(runner.dev_scores))

    for length, dev_score, test_score in zip(lengths, srn_dev_scores, srn_test_scores):
        print(f"[SRN] length:{length}, dev_score: {dev_score}, test_score: {test_score:.5f}")

    # ======SRN在不同长度的验证集和测试集数据上的表现，绘制成图片进行观察=======================

    plt.plot(lengths, srn_dev_scores, '-o', color='#e4007f', label="Dev Accuracy")
    plt.plot(lengths, srn_test_scores, '-o', color='#f19ec2', label="Test Accuracy")

    # 绘制坐标轴和图例
    plt.ylabel("accuracy", fontsize='large')
    plt.xlabel("sequence length", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    fig_name = "./images/6.7.pdf"
    fig_dir = os.path.dirname(fig_name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    #plt.savefig(fig_name)
    plt.show()