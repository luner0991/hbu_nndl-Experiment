'''
@功能: 使用PyTorch实现Seq2Seq编码器-解码器: 将输入的英语单词翻译成西班牙语
@作者: lxy
@日期: 2024/11/24
'''

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# S: 表示解码输入的开始符号
# E: 表示解码输出的结束符号
# ?: 用于填充空白序列，当当前批次的数据长度不足n_step时使用

letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
letter2idx = {n: i for i, n in enumerate(letter)}

# 示例数据: 英语单词与对应的西班牙语翻译
seq_data = [
    ['hello', 'hola'],  # 'hello' -> 'hola'（英语到西班牙语）
    ['cat', 'gato'],    # 'cat' -> 'gato'
    ['good', 'bueno'],  # 'good' -> 'bueno'
]

# Seq2Seq 参数
n_step = max([max(len(i), len(j)) for i, j in seq_data])  # 最大长度（=5）
n_hidden = 128  # 隐藏层维度
n_class = len(letter2idx)  # 类别数（即字符集大小）
batch_size = 3  # 批次大小

# 函数：构建训练数据
def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (n_step - len(seq[i]))  # 如 'man??', 'women'

        # 编码输入：将字符转为索引，并在末尾添加结束符'E'
        enc_input = [letter2idx[n] for n in (seq[0] + 'E')]  # ['m', 'a', 'n', '?', '?', 'E']
        # 解码输入：在开头添加起始符'S'
        dec_input = [letter2idx[n] for n in ('S' + seq[1])]  # ['S', 'w', 'o', 'm', 'e', 'n']
        # 解码输出：在末尾添加结束符'E'
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')]  # ['w', 'o', 'm', 'e', 'n', 'E']

        # 将每个输入转为独热编码
        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)  # 解码输出不进行独热编码

    # 返回Tensor格式数据
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.LongTensor(dec_output_all)

# 获取训练数据
enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)

# 自定义数据集
class TranslateDataSet(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self):  # 返回数据集的大小
        return len(self.enc_input_all)

    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]

# 数据加载器
loader = Data.DataLoader(TranslateDataSet(enc_input_all, dec_input_all, dec_output_all), batch_size, True)

# Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        # 编码器：RNN模型
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        # 解码器：RNN模型
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        # 全连接层，用于输出分类
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input：输入的编码数据 [batch_size, n_step+1, n_class]
        # dec_input：输入的解码数据 [batch_size, n_step+1, n_class]
        enc_input = enc_input.transpose(0, 1)  # 转置为 [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)  # 转置为 [n_step+1, batch_size, n_class]

        # 编码器输出：h_t 是最后的隐藏状态
        _, h_t = self.encoder(enc_input, enc_hidden)
        # 解码器输出：outputs 是解码过程中的所有输出
        outputs, _ = self.decoder(dec_input, h_t)

        # 通过全连接层输出最终结果
        model = self.fc(outputs)  # [n_step+1, batch_size, n_class]
        return model

# 实例化模型
model = Seq2Seq().to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练过程
for epoch in range(5000):
    for enc_input_batch, dec_input_batch, dec_output_batch in loader:
        # 初始化隐藏状态
        h_0 = torch.zeros(1, batch_size, n_hidden).to(device)

        # 将数据移至设备
        enc_input_batch, dec_input_batch, dec_output_batch = (
            enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device))

        # 训练模型，获取预测结果
        pred = model(enc_input_batch, h_0, dec_input_batch)

        # 计算损失
        pred = pred.transpose(0, 1)  # [batch_size, n_step+1, n_class]
        loss = 0
        for i in range(len(dec_output_batch)):
            loss += criterion(pred[i], dec_output_batch[i])  # 计算每一批次的损失

        # 每1000次迭代输出一次损失
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        # 反向传播并优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试函数：翻译单词
def translate(word):
    enc_input, dec_input, _ = make_data([[word, '?' * n_step]])
    enc_input, dec_input = enc_input.to(device), dec_input.to(device)
    hidden = torch.zeros(1, 1, n_hidden).to(device)  # 初始化隐藏状态
    output = model(enc_input, hidden, dec_input)

    # 获取最大概率的预测值
    predict = output.data.max(2, keepdim=True)[1]
    decoded = [letter[i] for i in predict]
    translated = ''.join(decoded[:decoded.index('E')])  # 直到'E'为止

    return translated.replace('?', '')  # 去掉填充符号

# 测试翻译效果
print('测试')
for seq in seq_data:
    word = seq[0]  # 获取英语单词
    translated_word = translate(word)  # 获取翻译结果
    print(f'{word} -> {translated_word}')
