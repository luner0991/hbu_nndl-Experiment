'''
@Function：基于双向LSTM实现文本分类
@Author: lxy
@Date: 2024/12/12
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from functools import partial
import random
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
数据集处理部分
'''
# 加载IMDB数据集
def load_imdb_data(path):
    """
        功能：从指定路径加载IMDB数据集，包括训练集、验证集和测试集
        参数：
            path: 数据集所在的路径
        返回值：
            trainset: 训练集数据，每条数据包含一个文本句子和对应的标签
            devset: 验证集数据
            testset: 测试集数据
        """
    assert os.path.exists(path)
    # 初始化数据集列表
    trainset, devset, testset = [], [], []
    # 加载训练集数据
    for label in ['pos', 'neg']:
        label_path = os.path.join(path, 'train', label)
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as f:
                    sentence = f.read().strip().lower()  # 读取并处理每个评论
                    trainset.append((sentence, label))

    # 加载测试集数据
    for label in ['pos', 'neg']:
        label_path = os.path.join(path, 'test', label)
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as f:
                    sentence = f.read().strip().lower()  # 读取并处理每个评论
                    testset.append((sentence, label))

    # 随机拆分测试集的一半作为验证集
    random.shuffle(testset)  # 打乱测试集顺序
    split_index = len(testset) // 2  # 计算拆分索引
    devset = testset[:split_index]  # 选择测试集前一半作为验证集
    testset = testset[split_index:]  # 剩下的部分作为测试集

    return trainset, devset, testset
# 加载IMDB数据集
train_data, dev_data, test_data = load_imdb_data("./dataset/")

# # 打印一下加载后的数据样式
# print(train_data[4])  # 打印训练集中的第5条数据


class IMDBDataset(Dataset):
    def __init__(self, examples, word2id_dict):
        super(IMDBDataset, self).__init__()
        self.word2id_dict = word2id_dict
        self.examples = self.words_to_id(examples)

    def words_to_id(self, examples):
        tmp_examples = []
        for idx, example in enumerate(examples):
            seq, label = example
            # 将单词映射为字典索引的ID， 对于词典中没有的单词用[UNK]对应的ID进行替代
            seq = [self.word2id_dict.get(word, self.word2id_dict['[UNK]']) for word in seq.split(" ")]

            # 映射标签: 'pos' -> 1, 'neg' -> 0
            label = 1 if label == 'pos' else 0  # 将标签从'pos'/'neg'转换为1/0
            tmp_examples.append([seq, label])
        return tmp_examples

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        return seq, label

    def __len__(self):
        return len(self.examples)

# # 加载词汇表文件，创建单词到ID的映射字典
def load_vocab(path):
    assert os.path.exists(path)  # 确保词表文件路径存在
    words = []  # 初始化空列表，存储词表中的单词
    with open(path, "r", encoding="utf-8") as f:  # 打开文件并读取内容
        words = f.readlines()  # 读取文件中的所有行
        words = [word.strip() for word in words if word.strip()]  # 移除每个单词的前后空白字符并去掉空行
    word2id = dict(zip(words, range(len(words))))  # 创建一个字典，将单词与对应的ID映射
    return word2id  # 返回这个字典


# 加载词表
word2id_dict = load_vocab("./dataset/imdb.vocab")
# 实例化Dataset
train_set = IMDBDataset(train_data, word2id_dict)
dev_set = IMDBDataset(dev_data, word2id_dict)
test_set = IMDBDataset(test_data, word2id_dict)

# print('训练集样本数：', len(train_set))
# print('样本示例：', train_set[4])


def collate_fn(batch_data, pad_val=0, max_seq_len=256):
    """
        整理批次数据，包括填充序列使其长度一致、将标签转换为张量等操作
        参数：
            batch_data: 包含一个批次数据的列表，每条数据包含一个词ID张量和对应的标签
            pad_val: 用于填充的数值，默认为1
        返回值：
            (seqs_tensor, lens_tensor), labels_tensor: 整理后的批次数据，包含填充后的序列张量、序列长度张量以及标签张量
        """
    seqs, seq_lens, labels = [], [], []
    max_len = 0
    for example in batch_data:
        seq, label = example
        # 对数据序列进行截断
        seq = seq[:max_seq_len]
        # 对数据截断并保存于seqs中
        seqs.append(seq)
        seq_lens.append(len(seq))
        labels.append(label)
        # 保存序列最大长度
        max_len = max(max_len, len(seq))
    # 对数据序列进行填充至最大长度
    for i in range(len(seqs)):
        seqs[i] = seqs[i] + [pad_val] * (max_len - len(seqs[i]))
    return (torch.tensor(seqs).to(device), torch.tensor(seq_lens)), torch.tensor(labels).to(device)
# =======测试==============
# max_seq_len = 5
# batch_data = [[[1, 2, 3, 4, 5, 6], 1], [[2, 4, 6], 0]]
# (seqs, seq_lens), labels = collate_fn(batch_data, pad_val=word2id_dict["[PAD]"], max_seq_len=max_seq_len)
# print("seqs: ", seqs)
# print("seq_lens: ", seq_lens)
# print("labels: ", labels)

# ===============封装dataloader=========================
max_seq_len = 256
batch_size = 128
collate_fn = partial(collate_fn, pad_val=word2id_dict["[PAD]"], max_seq_len=max_seq_len)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, drop_last=False, collate_fn=collate_fn)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size,
                                         shuffle=False, drop_last=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, drop_last=False, collate_fn=collate_fn)
'''
数据集处理部分结束
'''

'''
模型构建部分
'''
# ======================汇聚层====================
class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()

    def forward(self, sequence_output, sequence_length):
        # 假设 sequence_length 是一个 PyTorch 张量
        sequence_length = sequence_length.unsqueeze(-1).to(torch.float32)
        # 根据sequence_length生成mask矩阵，用于对Padding位置的信息进行mask
        max_len = sequence_output.shape[1]

        mask = torch.arange(max_len, device='cuda') < sequence_length.to('cuda')
        mask = mask.to(torch.float32).unsqueeze(-1)
        # 对序列中paddling部分进行mask

        sequence_output = torch.multiply(sequence_output, mask.to('cuda'))
        # 对序列中的向量取均值
        batch_mean_hidden = torch.divide(torch.sum(sequence_output, dim=1), sequence_length.to('cuda'))
        return batch_mean_hidden
# ===================模型汇总=====================
class Model_BiLSTM_FC(nn.Module):
    def __init__(self, num_embeddings, input_size, hidden_size, num_classes=2):
        super(Model_BiLSTM_FC, self).__init__()
        # 词典大小
        self.num_embeddings = num_embeddings
        # 单词向量的维度
        self.input_size = input_size
        # LSTM隐藏单元数量
        self.hidden_size = hidden_size
        # 情感分类类别数量
        self.num_classes = num_classes
        # 实例化嵌入层
        self.embedding_layer = nn.Embedding(num_embeddings, input_size, padding_idx=0)
        # 实例化LSTM层
        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # 实例化聚合层
        self.average_layer = AveragePooling()
        # 实例化输出层
        self.output_layer = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs):
        # 对模型输入拆分为序列数据和mask
        input_ids, sequence_length = inputs
        # 获取词向量
        inputs_emb = self.embedding_layer(input_ids)

        packed_input = nn.utils.rnn.pack_padded_sequence(inputs_emb, sequence_length.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        # 使用lstm处理数据
        packed_output, _ = self.lstm_layer(packed_input)
        # 解包输出
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # 使用聚合层聚合sequence_output
        batch_mean_hidden = self.average_layer(sequence_output, sequence_length)
        # 输出文本分类logits
        logits = self.output_layer(batch_mean_hidden)
        return logits
'''
模型构建部分结束
'''
'''
模型训练部分
'''
# ===============模型训练===================
from Runner import RunnerV3,Accuracy,plot_training_loss_acc
np.random.seed(0)
random.seed(0)
torch.seed()

# 指定训练轮次
num_epochs = 3
# 指定学习率
learning_rate = 0.001
# 指定embedding的数量为词表长度
num_embeddings = len(word2id_dict)
# embedding向量的维度
input_size = 256
# LSTM网络隐状态向量的维度
hidden_size = 256
# 模型保存目录
save_dir = "./checkpoints/best.pdparams"

# 实例化模型
model = Model_BiLSTM_FC(num_embeddings, input_size, hidden_size).to(device)
# 指定优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# 指定损失函数
loss_fn = nn.CrossEntropyLoss()
# 指定评估指标
metric = Accuracy()
# 实例化Runner
runner = RunnerV3(model, optimizer, loss_fn, metric)
# 模型训练
start_time = time.time()
runner.train(train_loader, dev_loader, num_epochs=num_epochs, eval_steps=10, log_steps=10,
             save_path=save_dir)
end_time = time.time()
print("time: ", (end_time - start_time))
# ============  绘制训练过程中在训练集和验证集上的损失图像和在验证集上的准确率图像 ===========
# sample_step: 训练损失的采样step，即每隔多少个点选择1个点绘制
# loss_legend_loc: loss 图像的图例放置位置
# acc_legend_loc： acc 图像的图例放置位置
plot_training_loss_acc(runner,  fig_size=(16, 6), sample_step=10, loss_legend_loc="lower left",
                       acc_legend_loc="lower right")
'''
模型训练部分结束
'''

# ==================== 模型评价 =============
model_path = "./checkpoints/best.pdparams"
runner.load_model(model_path)
accuracy, _ =  runner.evaluate(test_loader)
print(f"Evaluate on test set, Accuracy: {accuracy:.5f}")

# =====================模型预测==========
id2label={0:"消极情绪", 1:"积极情绪"}
text = "this movie is so great. I watched it three times already"
# 处理单条文本
sentence = text.split(" ")
words = [word2id_dict[word] if word in word2id_dict else word2id_dict['[UNK]'] for word in sentence]
words = words[:max_seq_len]
sequence_length = torch.tensor([len(words)], dtype=torch.int64)
words = torch.tensor(words, dtype=torch.int64).unsqueeze(0)
# 使用模型进行预测
logits = runner.predict((words.to(device), sequence_length.to(device)))
max_label_id = torch.argmax(logits, dim=-1).cpu().numpy()[0]
pred_label = id2label[max_label_id]
print("Label: ", pred_label)