import torch


def accuracy(preds, labels):
    """
    输入：
        - preds：预测值，二分类时，shape=[N, 1]，N为样本数量，多分类时，shape=[N, C]，C为类别数量
        - labels：真实标签，shape=[N, 1]
    输出：
        - 准确率：shape=[1]
    """
    # 判断是二分类任务还是多分类任务，preds.shape[1]=1时为二分类任务，preds.shape[1]>1时为多分类任务
    if preds.shape[1] == 1:
        # 二分类时，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0
        # 使用 'torch.round' 进行四舍五入，将概率值转换为二进制标签
        preds = torch.round(preds)
    else:
        # 多分类时，使用 'torch.argmax' 计算最大元素索引作为类别
        preds = torch.argmax(preds, dim=1)

    # 计算准确率
    correct = (preds == labels).sum().item()
    #     print("correct:",correct)
    accuracy = correct / len(labels)
    #     print("shape of labels:",labels.shape)
    #     print("labels:",labels)
    #     print("shape of preds:",preds.shape)
    #     print("preds:",preds)
    return accuracy


# 假设模型的预测值为[[0.],[1.],[1.],[0.]]，真实类别为[[1.],[1.],[0.],[0.]]，计算准确率
preds = torch.tensor([[0.], [1.], [1.], [0.]])
labels = torch.tensor([[1.], [1.], [0.], [0.]])
print("accuracy is:", accuracy(preds, labels))