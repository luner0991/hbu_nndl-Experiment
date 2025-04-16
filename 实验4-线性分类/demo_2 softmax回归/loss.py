import torch
import torch.nn as nn

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, C]，N为样本数量，C为类别数量
            - labels：真实标签，shape=[N]
        输出：
            - 损失值：shape=[1]
        """
        # 将标签转换为长整型
        labels = labels.view(-1).long()
        N = predicts.shape[0]  # 样本数量
        loss = 0.0

        # 计算损失
        for i in range(N):
            index = labels[i]  # 获取当前样本的标签
            loss -= torch.log(predicts[i][index])  # 计算交叉熵损失

        return loss / N  # 返回平均损失

# 测试一下
# 假设真实标签为第0类
labels = torch.tensor([0, 1, 0])  # 真实标签（3个样本）
# 假设的预测值（3个样本，2个类别）
outputs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

# 计算损失函数
mce_loss = MultiCrossEntropyLoss()
loss = mce_loss(outputs, labels)
print(loss)
