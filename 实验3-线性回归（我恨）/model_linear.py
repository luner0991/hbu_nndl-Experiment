import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        # 初始化模型参数
        self.params = {}
        self.params['w'] = nn.Parameter(torch.randn(size=(input_size, 1)))  # 权重初始化为随机值
        self.params['b'] = nn.Parameter(torch.zeros([1]))  # 偏置初始化为零

    def forward(self, X, y_true=None):
        """
        前向传播函数
        输入：
           - X: tensor, shape=[N, D]，N 为样本数量，D 为特征维度
           - y_true: tensor, shape=[N, 1]，真实标签（可选）
        输出：
           - y_pred: tensor, shape=[N, 1]，模型的预测值
           - mse_loss: 均方损失（如果提供了真实标签）
        """
        N, D = X.shape
        assert D == self.params['w'].shape[0], "输入维度与模型参数不匹配"
        y_pred = torch.matmul(X, self.params['w']) + self.params['b']

        # 计算均方损失
        mse_loss = None
        if y_true is not None:
            error = y_pred - y_true
            mse_loss = torch.mean(error ** 2)  # 计算均方损失
        return y_pred, mse_loss  # 返回预测值和损失（如果有）

# 设置输入数据的维度
input_size = 3
N = 2
# 生成 2 个维度为 3 的随机数据
X = torch.randn(size=[N, input_size], dtype=torch.float32)
print("输入：", X)
# 实例化线性模型
model = LinearModel(input_size)
# 进行前向传播计算
y_pred, _ = model(X)  # 只获取预测值
# 生成随机目标值，作为真实标签
y_true = torch.randn(size=[N, 1], dtype=torch.float32)
print("真实标签：", y_true)  # 打印真实标签
# 进行前向传播计算并计算损失
y_pred, mse_loss = model(X, y_true)
# 打印输出
print("预测值:", y_pred)  # 打印预测值
print("均方损失:", mse_loss.item())  # 打印损失值
