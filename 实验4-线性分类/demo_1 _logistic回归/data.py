import math
import copy
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def make_moons(n_samples=1000, shuffle=True, noise=None):
    """
    生成带噪音的弯月形状数据
    输入：
        - n_samples：数据量大小，数据类型为int
        - shuffle：是否打乱数据，数据类型为bool
        - noise：以多大的程度增加噪声，数据类型为None或float，noise为None时表示不增加噪声
    输出：
        - X：特征数据，shape=[n_samples,2]
        - y：标签数据, shape=[n_samples]
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    # 采集第1类数据，特征为(x,y)
    # 使用'torch.linspace'在0到pi上均匀取n_samples_out个值
    # 使用'torch.cos'计算上述取值的余弦值作为特征1，使用'torch.sin'计算上述取值的正弦值作为特征2
    outer_circ_x = torch.cos(torch.linspace(0, math.pi, n_samples_out))
    outer_circ_y = torch.sin(torch.linspace(0, math.pi, n_samples_out))
    inner_circ_x = 1 - torch.cos(torch.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_samples_in))
    print('外弯月特征x的形状:', outer_circ_x.shape, '外弯月特征y的形状:', outer_circ_y.shape)
    print('内弯月特征x的形状:', inner_circ_x.shape, '内弯月特征y的形状:', inner_circ_y.shape)

    # 使用'torch.cat'将两类数据的特征1和特征2分别沿维度0拼接在一起，得到全部特征1和特征2
    # 使用'torch.stack'将两类特征沿维度1堆叠在一起
    X = torch.stack(
        [torch.cat([outer_circ_x, inner_circ_x]),
         torch.cat([outer_circ_y, inner_circ_y])],
        axis=1
    )

    print('拼接后的形状:', torch.cat([outer_circ_x, inner_circ_x]).shape)
    print('X的形状:', X.shape)

    # 使用'torch.zeros'将第一类数据的标签全部设置为0
    # 使用'torch.ones'将第二类数据的标签全部设置为1
    y = torch.cat(
        [torch.zeros(size=[n_samples_out]), torch.ones(size=[n_samples_in])]
    )

    print('y的形状:', y.shape)

    # 如果shuffle为True，将所有数据打乱
    if shuffle:
        # 使用'torch.randperm'生成一个数值在0到X.shape[0]，随机排列的一维Tensor作为索引值，用于打乱数据
        print(X.shape[0])

        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    # 如果noise不为None，则给特征值加入噪声
    if noise is not None:
        # 使用'torch.normal'生成符合正态分布的随机Tensor作为噪声，并加到原始特征上
        print(noise)
        X += torch.normal(mean=0.0, std=noise, size=X.shape)

    return X, y

# 采样1000个样本
n_samples = 1000
X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.2)

# 可视化生成的数据集，不同颜色代表不同类别
plt.figure(figsize=(5,5))
plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
plt.xlim(-3,4)
plt.ylim(-3,4)
plt.savefig('线性数据集可视化.pdf')
plt.show()

num_train = 640
num_dev = 160
num_test = 200
X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]
y_train = y_train.reshape([-1,1])
y_dev = y_dev.reshape([-1,1])
y_test = y_test.reshape([-1,1])
# 打印X_train和y_train的维度
print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)