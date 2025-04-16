import torch

torch.seed()  # 设置随机种子
class Op(object):
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

def optimizer_lsm(model, X, y, reg_lambda=0):
    N, D = X.shape
    # 对输入特征数据所有特征向量求平均
    x_bar_tran = torch.mean(X, axis=0).T
    # 求标签的均值,shape=[1]
    y_bar = torch.mean(y)
    # torch.subtract通过广播的方式实现矩阵减向量
    x_sub = torch.subtract(X, x_bar_tran)
    # 使用torch.all判断输入tensor是否全0
    if torch.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = torch.zeros([D])
        return model
    # torch.inverse求方阵的逆
    tmp = torch.inverse(torch.matmul(x_sub.T, x_sub) +
                         reg_lambda * torch.eye(D))

    w = torch.matmul(torch.matmul(tmp, x_sub.T), (y - y_bar))

    b = y_bar - torch.matmul(x_bar_tran, w)

    model.params['b'] = b
    model.params['w'] = torch.squeeze(w, axis=-1)

    return model
