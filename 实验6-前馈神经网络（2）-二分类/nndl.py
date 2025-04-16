import torch
from data import make_moons
import Runner2
torch.seed()  # 设置随机种子
class Op(object):
    """
    基类，用于定义操作（如线性层、激活函数等）。

    方法：
        - __call__(inputs): 调用 forward 方法。
        - forward(inputs): 前向传播，必须在子类中实现。
        - backward(inputs): 反向传播，必须在子类中实现。
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError
# 线性算子
class Linear(Op):
    def __init__(self, input_size):

        self.input_size = input_size

        # 模型参数
        self.params = {}
        self.params['w'] = torch.randn(self.input_size, 1)
        self.params['b'] = torch.zeros([1])

    def __call__(self, X):
        return self.forward(X)

    # 前向函数
    def forward(self, X):
        N, D = X.shape

        if self.input_size == 0:
            return torch.full([N, 1], fill_value=self.params['b'])

        assert D == self.input_size  # 输入数据维度合法性验证

        # 使用torch.matmul计算两个tensor的乘积
        y_pred = torch.matmul(X, self.params['w']) + self.params['b']
        return y_pred

class Optimizer(Op):
    """
    优化器基类，用于更新模型参数。

    参数：
        - model: 需要优化的模型
        - init_lr: 初始学习率

    方法：
        - step(): 执行一步优化，更新模型参数。
        - zero_grad(): 清零模型的梯度。
    """
    def __init__(self, model, init_lr):
        self.model = model
        self.init_lr = init_lr

    def step(self):
        raise NotImplementedError("必须在子类中实现该方法。")

    def zero_grad(self):
        for layer in self.model.layers:
            if hasattr(layer, 'grads'):
                for key in layer.grads.keys():
                    layer.grads[key] = torch.zeros_like(layer.grads[key])  # 清零梯度
def accuracy(preds, labels):
    """
    计算模型预测的准确率。

    参数：
        - preds: 预测值，shape=[N, 1]（二分类）或 [N, C]（多分类）
        - labels: 真实标签，shape=[N, 1]

    返回：
        - 准确率：float
    """
    if preds.shape[1] == 1:
        preds = torch.round(preds)  # 二分类，四舍五入
    else:
        preds = torch.argmax(preds, dim=1)  # 多分类，获取最大元素索引

    correct = (preds == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy