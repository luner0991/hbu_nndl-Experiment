'''
@Function：基于双向LSTM和注意力机制实现文本分类
@Author: lxy
@Date: 2024/12/12
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial
import random
import numpy as np
import time
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
    添加全局配置，默认放到cuda上
"""
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # 设置默认张量类型为GPU上的FloatTensor类型
    print("放置成功")
else:
    torch.set_default_tensor_type('torch.FloatTensor')  # 如果GPU不可用，设置为CPU上的FloatTensor类型
# seqs是一个由tensor格式的元素组成的列表，将这个列表转为tensor
def list2tensor(seqs):
    # 将seqs中的每个Tensor元素转换为numpy数组
    numpy_seqs = [element.detach().cpu().numpy() for element in seqs]
    # 将numpy数组沿着某个轴（这里假设沿着第一个轴，类似torch.stack里的dim=0的效果，可根据实际调整）进行堆叠
    stacked_numpy_seqs = np.stack(numpy_seqs, axis=0)
    # 再将堆叠后的numpy数组转换为Tensor并移动到指定设备上
    seqs_tensor = torch.from_numpy(stacked_numpy_seqs).to(device)
    return seqs_tensor

# 加载IMDB数据集
def load_imdb_data(path):
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
    random.shuffle(trainset)
    split_index = len(testset) // 2  # 计算拆分索引
    devset = testset[:split_index]  # 选择测试集前一半作为验证集
    testset = testset[split_index:]  # 剩下的部分作为测试集

    return trainset, devset, testset

# 加载词汇表文件，创建单词到ID的映射字典
def load_vocab(path):
    """
        功能：从给定路径加载词汇表文件，并创建单词到ID的映射字典
        参数：
            path: 词汇表文件的路径
        返回值：
            word2id: 单词到ID的映射字典，形如 {'word1': 0, 'word2': 1,...}
        """
    assert os.path.exists(path)  # 确保词表文件路径存在
    words = []  # 初始化空列表，存储词表中的单词
    with open(path, "r", encoding="utf-8") as f:  # 打开文件并读取内容
        words = f.readlines()  # 读取文件中的所有行
        words = [word.strip() for word in words if word.strip()]  # 移除每个单词的前后空白字符并去掉空行
    word2id = dict(zip(words, range(len(words))))  # 创建一个字典，将单词与对应的ID映射
    return word2id  # 返回这个字典

# 自定义数据集类，用于处理IMDB数据集中的文本和标签
class IMDBDataset(Dataset):
    def __init__(self, examples, word2id_dict):
        super(IMDBDataset, self).__init__()
        self.word2id_dict = word2id_dict
        self.examples = self.words_to_id(examples)
    '''
          将文本句子中的单词转换为对应的ID，处理不在词汇表中的单词（用[UNK]对应的ID替代），并将标签转换为数值形式（'pos' -> 1, 'neg' -> 0）
    '''
    def words_to_id(self, examples):
        tmp_examples = []
        for idx, example in enumerate(examples):
            seq, label = example
            # 将单词映射为字典索引的ID， 对于词典中没有的单词用[UNK]对应的ID进行替代
            seq = [self.word2id_dict.get(word, self.word2id_dict['[UNK]']) for word in seq.split(" ")]
            seq_tensor = torch.tensor(seq).to(device)  # 将张量移到正确设备
            # 映射标签: 'pos' -> 1, 'neg' -> 0
            label = 1 if label == 'pos' else 0  # 将标签从'pos'/'neg'转换为1/0
            tmp_examples.append([seq_tensor, label])
        return tmp_examples

    def __getitem__(self, idx): # 获取指定索引位置的示例数据
        seq, label = self.examples[idx]
        return seq, label

    def __len__(self): # 返回数据集的大小
        return len(self.examples)


def collate_fn(batch_data, pad_val=1):
    seqs = []
    labels = []
    lens = []
    for seq, label in batch_data:
        seqs.append(seq)
        labels.append(label)  # 这里不需要再额外嵌套一层列表，直接添加label即可
        lens.append(len(seq))

    max_len = max(lens)
    # 对序列进行填充操作，使其长度统一为批次中的最大长度
    for i in range(len(seqs)):
        try:
            # 将填充部分转换为Tensor，形状需要和seqs[i]的除了序列长度维度外其他维度匹配，这里假设seqs[i]是1维序列，所以形状为 (需要填充的长度,)
            padding_tensor = torch.full((max_len - len(seqs[i]),), pad_val, dtype=torch.long).to(device)
            # 使用torch.cat进行拼接，在维度0（序列长度维度）上拼接
            seqs[i] = torch.cat([seqs[i], padding_tensor], dim=0)
        except:
            print(f"出现异常，当前seqs[{i}]的形状为: {seqs[i].shape if isinstance(seqs[i], torch.Tensor) else None}")
            print(f"填充部分 [pad_val] * (max_len - len(seqs[i])) 的形状理论上应为 ({max_len - len(seqs[i])},)，实际类型为: {type([pad_val] * (max_len - len(seqs[i])))}")
            break

    # 将序列列表转换为PyTorch张量
    seqs_tensor = list2tensor(seqs).to(device)
    lens_tensor = torch.tensor(lens).to(device)
    labels_tensor = torch.tensor(labels).unsqueeze(1).to(device)
    return (seqs_tensor, lens_tensor), labels_tensor

# ===============封装dataloader=========================
 # 加载IMDB数据集
train_data, dev_data, test_data = load_imdb_data("./dataset/")
# 加载词汇表字典
word2id_dict = load_vocab("./dataset/imdb.vocab")
# 实例化Dataset
train_set = IMDBDataset(train_data, word2id_dict)
dev_set = IMDBDataset(dev_data, word2id_dict)
test_set = IMDBDataset(test_data, word2id_dict)
max_seq_len = 256
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=False, drop_last=False, collate_fn=collate_fn)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size,

                                         shuffle=False, drop_last=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, drop_last=False, collate_fn=collate_fn)



'''
注意力打分函数
'''
# ===============加性==================
class AdditiveScore(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveScore, self).__init__()
        self.fc_W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_v = nn.Linear(hidden_size, 1, bias=False)
        # 查询向量使用均匀分布随机初始化
        self.q = torch.rand(1, hidden_size) * (0.5 - (-0.5)) + (-0.5)  # 使用torch.rand生成[0, 1]均匀分布随机数并转换到[-0.5, 0.5]范围

    def forward(self, inputs):
        """
        输入：
            - inputs：输入矩阵，shape=[batch_size, seq_len, hidden_size]
        输出：
            - scores：输出矩阵，shape=[batch_size, seq_len]
        """
        # inputs:  [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = inputs.shape
        # 将查询向量扩展到与输入批次维度匹配，方便后续计算
        q_expanded = self.q.expand(batch_size, -1, hidden_size)
        # scores: [batch_size, seq_len, hidden_size]
        scores = torch.tanh(self.fc_W(inputs) + self.fc_U(q_expanded))
        # scores: [batch_size, seq_len]
        scores = self.fc_v(scores).squeeze(-1)
        return scores

# torch.manual_seed(2021)
# inputs = torch.rand(1, 3, 3)
# additiveScore = AdditiveScore(hidden_size=3)
# scores = additiveScore(inputs)
# print(scores)

# =================点积========================
class DotProductScore(nn.Module):
    def __init__(self, hidden_size):
        super(DotProductScore, self).__init__()
        # 使用均匀分布随机初始化一个查询向量
        self.q = torch.rand(hidden_size, 1) * (0.5 - (-0.5)) + (-0.5)  # 通过torch.rand生成[0, 1]均匀分布随机数并转换到[-0.5, 0.5]范围

    def forward(self, inputs):
        """
        输入：
            - inputs：输入矩阵，shape=[batch_size, seq_len, hidden_size]
        输出：
            - scores：输出矩阵，shape=[batch_size, seq_len]
        """
        # 获取输入张量的形状信息
        batch_size, seq_length, hidden_size = inputs.shape
        # 进行点积运算，在PyTorch中使用torch.matmul实现
        scores = torch.matmul(inputs, self.q)
        # 去除最后一维（维度大小为1的那一维）
        scores = scores.squeeze(-1)
        return scores


# torch.manual_seed(2021)
# inputs = torch.rand(1, 3, 3)
# dotScore = DotProductScore(hidden_size=3)
# scores = dotScore(inputs)
# print(scores)

'''
注意力层
'''
class Attention(nn.Module):
    def __init__(self, hidden_size, use_additive=False):
        super(Attention, self).__init__()
        self.use_additive = use_additive
        # 使用加性模型或者点积模型
        if self.use_additive:
            self.scores = AdditiveScore(hidden_size)
        else:
            self.scores = DotProductScore(hidden_size)
        self._attention_weights = None

    def forward(self, X, valid_lens):
        """
        输入：
            - X：输入矩阵，shape=[batch_size, seq_len, hidden_size]
            - valid_lens：长度矩阵，shape=[batch_size]
        输出：
            - context ：输出矩阵，表示的是注意力的加权平均的结果
        """
        # scores: [batch_size, seq_len]
        scores = self.scores(X)
        # arrange: [1,seq_len],比如seq_len=4, arrange变为 [0,1,2,3]
        arrange = torch.arange(scores.shape[1], dtype=torch.float32).unsqueeze(0)
        # valid_lens : [batch_size, 1]
        valid_lens = valid_lens.unsqueeze(1)
        # mask [batch_size, seq_len]
        mask = arrange < valid_lens
        y = torch.full(scores.shape, float('-1e9'), dtype=torch.float32)
        scores = torch.where(mask, scores, y)
        # attn_weights: [batch_size, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        self._attention_weights = attn_weights
        # context: [batch_size, 1, hidden_size]
        context = torch.matmul(attn_weights.unsqueeze(1), X)
        # context: [batch_size, hidden_size]
        context = context.squeeze(1)
        return context

    @property
    def attention_weights(self):
        return self._attention_weights

# 使用不同的打分函数进行测试
# torch.manual_seed(2021)
# X = torch.rand(1, 3, 3)
# valid_lens = torch.tensor([2])
# print("输入向量为 {}".format(X))
# add_atten = Attention(hidden_size=3, use_additive=True)
# context1 = add_atten(X, valid_lens)
# dot_atten = Attention(hidden_size=3, use_additive=False)
# context2 = dot_atten(X, valid_lens)
# print("============使用加性打分函数时:=============")
# print("注意力的输出为 : {}".format(context1))
# print("注意力权重为 : {}".format(add_atten.attention_weights))
# print("============使用点积打分函数时:=============")
# print("注意力的输出为 : {}".format(context2))
# print("注意力权重为 : {}".format(dot_atten.attention_weights))

'''
模型汇总
'''
class Model_LSTMAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size,
        vocab_size,
        n_classes=10,
        n_layers=1,
        use_additive=False,
    ):
        super(Model_LSTMAttention, self).__init__()
        # 表示LSTM单元的隐藏神经元数量，它也将用来表示hidden和cell向量状态的维度
        self.hidden_size = hidden_size
        # 表示词向量的维度
        self.embedding_size = embedding_size
        # 表示词典的的单词数量
        self.vocab_size = vocab_size
        # 表示文本分类的类别数量
        self.n_classes = n_classes
        # 表示LSTM的层数
        self.n_layers = n_layers
        # 定义embedding层
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_size
        )
        # 定义LSTM，它将用来编码网络
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=True,  # 在PyTorch中使用bidirectional=True表示双向LSTM
        )
        # lstm的维度输出
        output_size = self.hidden_size * 2

        # 定义Attention层
        self.attention = Attention(output_size, use_additive=use_additive)

        # 定义分类层，用于将语义向量映射到相应的类别
        self.cls_fc = nn.Linear(
            in_features=output_size, out_features=self.n_classes
        )

    def forward(self, inputs):
        input_ids, valid_lens = inputs
        # 获取训练的batch_size
        batch_size = input_ids.shape[0]
        # 获取词向量并且进行dropout（这里代码中没体现dropout操作，若后续需要可添加）
        embedded_input = self.embedding(input_ids)
        # 使用LSTM进行语义编码
        # 在PyTorch中，LSTM的输入参数顺序有所不同，同时不需要显式传入sequence_length参数（它会在内部处理序列长度相关情况）
        output, (last_hidden, last_cell) = self.lstm(embedded_input)
        # 使用注意力机制
        # 进行Attention, attn_weights: [batch_size, seq_len]
        output = self.attention(output, valid_lens)
        # 将其通过分类线性层，获得初步的类别数值
        logits = self.cls_fc(output)
        return logits
'''
模型训练    
'''
from Runner import RunnerV3,Accuracy,plot_training_loss_acc
# 迭代的epoch
epochs = 2
# 词汇表的大小
vocab_size = len(word2id_dict)
# lstm的输出单元的大小
hidden_size = 128
# embedding的维度
embedding_size = 128
# 类别数
n_classes = 2
# lstm的层数
n_layers = 1
# 学习率
learning_rate = 0.001
# 定义交叉熵损失
criterion = nn.CrossEntropyLoss()
# 指定评价指标
metric = Accuracy()
# 实例化基于LSTM的注意力模型
model_atten = Model_LSTMAttention(
    hidden_size,
    embedding_size,
    vocab_size,
    n_classes=n_classes,
    n_layers=n_layers,
    use_additive=True,
)
model_atten.to(device)


# 定义优化器
optimizer = torch.optim.Adam(model_atten.parameters(), lr=learning_rate)
# 实例化
runner = RunnerV3(model_atten, optimizer, criterion, metric)
save_path = "./checkpoint2/model_best.pth"
start_time = time.time()
# 训练，假设train_loader和dev_loader是已经正确定义好的PyTorch的DataLoader对象，用于加载训练集和验证集数据
runner.train(
    train_loader,
    dev_loader,
    num_epochs=epochs,
    log_steps=10,
    eval_steps=10,
    save_path=save_path,
)
end_time = time.time()
print("训练时间:{}".format(end_time - start_time))

# #plot_training_loss_acc(runner,  fig_size=(16, 6), sample_step=10, loss_legend_loc="lower left",
#                        acc_legend_loc="lower right")
'''
模型评价
'''
model_path = "./checkpoint2/model_best.pth"
runner.load_model(model_path)
accuracy, _ =  runner.evaluate(test_loader)
print(f"Evaluate on test set, Accuracy: {accuracy:.5f}")

# ============注意力可视化===============
model_path = "checkpoint2/model_best.pth"  # 将后缀修改为.pth，这是PyTorch模型保存的常见后缀格式
model_atten = Model_LSTMAttention(
    hidden_size,
    embedding_size,
    vocab_size,
    n_classes=n_classes,
    n_layers=n_layers,
    use_additive=True,
)
model_state_dict = torch.load(model_path)  # 使用torch.load加载模型参数
model_atten.load_state_dict(model_state_dict)  # 将加载的参数应用到模型中
text = "this great science fiction film is really awesome"
# text = "This movie was craptacular"
# text = "I got stuck in traffic on the way to the theater"
# 分词
sentence = text.split(" ")
# 词映射成ID的形式
tokens = [
    word2id_dict[word] if word in word2id_dict else word2id_dict["[oov]"]
    for word in sentence
]
# 取前max_seq_len的单词
tokens = tokens[:max_seq_len]
# 序列长度
seq_len = torch.tensor([len(tokens)], dtype=torch.long)  # 将序列长度转为torch的LongTensor类型
# 转换成PyTorch的Tensor，并增加一个批次维度（unsqueeze(0)的作用）
input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # 确保移到相应设备上（如GPU）
inputs = [input_ids, seq_len]
# 模型开启评估模式
model_atten.eval()
# 设置不求梯度，使用torch.no_grad上下文管理器
with torch.no_grad():
    # 预测输出
    pred_prob = model_atten(inputs)
# 提取注意力权重
atten_weights = model_atten.attention.attention_weights
print("输入的文本为：{}".format(text))
print("转换成id的形式为：{}".format(input_ids.cpu().numpy()))  # 将张量移到CPU上再转为numpy数组进行打印输出
print("训练的注意力权重为：{}".format(atten_weights.cpu().numpy()))  # 同样移到CPU上转为numpy数组打印

import seaborn as sns
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# 对文本进行分词，得到过滤后的词
list_words = text.split(" ")
# 移到CPU上转为numpy数组
atten_weights = atten_weights.cpu().numpy()
# 提取注意力权重，转换成list
data_attention = atten_weights.tolist()
print(data_attention)
# 取出前max_seq_len变换进行特征融合，得到最后个词
list_words = list_words[:max_seq_len]
# 把权重转换为DataFrame，列名为单词
d = pd.DataFrame(data=data_attention, columns=list_words)
f, ax = plt.subplots(figsize=(20, 1.5))
sns.heatmap(d, vmin=0, vmax=0.4, ax=ax, cmap="OrRd")
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment="right")
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=0, horizontalalignment="right", fontsize=20)
plt.show()
