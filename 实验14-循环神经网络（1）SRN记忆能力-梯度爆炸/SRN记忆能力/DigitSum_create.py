
import random
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
import torch.nn.functional as F
import time
from Runner import Accuracy, RunnerV3
import matplotlib.pyplot as plt

# 固定随机种子
random.seed(0)
np.random.seed(0)

def generate_data(length, k, save_path):
    if length < 3:
        raise ValueError("The length of data should be greater than 2.")
    if k == 0:
        raise ValueError("k should be greater than 0.")
    # 生成100条长度为length的数字序列，除前两个字符外，序列其余数字暂用0填充
    base_examples = []
    for n1 in range(0, 10):
        for n2 in range(0, 10):
            seq = [n1, n2] + [0] * (length - 2)
            label = n1 + n2
            base_examples.append((seq, label))

    examples = []
    # 数据增强：对base_examples中的每条数据，默认生成k条数据，放入examples
    for base_example in base_examples:
        for _ in range(k):
            # 随机生成替换的元素位置和元素
            idx = np.random.randint(2, length)
            val = np.random.randint(0, 10)
            # 对序列中的对应零元素进行替换
            seq = base_example[0].copy()
            label = base_example[1]
            seq[idx] = val
            examples.append((seq, label))

    # 保存增强后的数据
    with open(save_path, "w", encoding="utf-8") as f:
        for example in examples:
            # 将数据转为字符串类型，方便保存
            seq = [str(e) for e in example[0]]
            label = str(example[1])
            line = " ".join(seq) + "\t" + label + "\n"
            f.write(line)

    print(f"generate data to: {save_path}.")


# 定义生成的数字序列长度
lengths = [5, 10, 15, 20, 25, 30, 35]
for length in lengths:
    # 构建数据集的基础目录路径（如果不存在就创建）
    base_dir = f"./datasets/{length}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # 生成长度为length的训练数据
    train_save_path = f"./datasets/{length}/train.txt"
    k = 3
    generate_data(length, k, train_save_path)
    # 生成长度为length的验证数据
    dev_save_path = f"./datasets/{length}/dev.txt"
    k = 1
    generate_data(length, k, dev_save_path)
    # 生成长度为length的测试数据
    test_save_path = f"./datasets/{length}/test.txt"
    k = 1
    generate_data(length, k, test_save_path)
