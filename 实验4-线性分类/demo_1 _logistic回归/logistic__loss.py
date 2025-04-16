import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.predicts = None
        self.labels = None
        self.num = None

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, 1]，N为样本数量
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]

        # 计算二元交叉熵损失
        loss = -1. / self.num * (
                    torch.matmul(self.labels.t(), torch.log(self.predicts)) + torch.matmul((1 - self.labels.t()),
                                                                                           torch.log(
                                                                                               1 - self.predicts)))
        loss = torch.squeeze(loss, axis=1)
        return loss


# 测试一下
# 生成一组长度为3，值为1的标签数据
labels = torch.ones(size=[3, 1])
# 假设outputs是模型的输出
outputs = torch.rand(size=[3, 1])  # 随机生成模型的输出
# 计算损失
bce_loss = BinaryCrossEntropyLoss()
loss = bce_loss(outputs, labels)
print(loss.item())  # 打印损失值