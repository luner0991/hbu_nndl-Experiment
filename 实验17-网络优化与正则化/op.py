import torch
from abc import abstractmethod


class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    # 输入：张量inputs
    # 输出：张量outputs
    def forward(self, inputs):
        raise NotImplementedError

    # 输入：最终输出对outputs的梯度outputs_grads
    # 输出：最终输出对inputs的梯度inputs_grads
    def backward(self, outputs_grads):
        raise NotImplementedError


# 优化器基类
class Optimizer(object):
    def __init__(self, init_lr, model):
        """
        优化器类初始化
        """
        self.init_lr = init_lr  # 初始化学习率
        self.model = model  # 模型对象

    @abstractmethod
    def step(self):
        """
        定义每次迭代如何更新参数
        """
        pass


class SimpleBatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(SimpleBatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        # 参数更新
        if isinstance(self.model.params, dict):
            for key in self.model.params.keys():
                self.model.params[key] -= self.init_lr * self.model.grads[key]


class BatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(BatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        # 参数更新
        for layer in self.model.layers:  # 遍历所有层
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] -= self.init_lr * layer.grads[key]


class Linear(Op):
    def __init__(self, in_features, out_features, name, weight_init=torch.randn, bias_init=torch.zeros):
        self.params = {}
        self.params['W'] = weight_init([in_features, out_features])
        self.params['b'] = bias_init([1, out_features])

        self.inputs = None
        self.grads = {}

        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        outputs = torch.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs

    def backward(self, grads):
        """
        输入：
            - grads：损失函数对当前层输出的导数
        输出：
            - 损失函数对当前层输入的导数
        """
        self.grads['W'] = torch.matmul(self.inputs.T, grads)
        self.grads['b'] = torch.sum(grads, dim=0)
        return torch.matmul(grads, self.params['W'].T)


class Logistic(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + torch.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self, outputs_grads):
        # 计算Logistic激活函数对输入的导数
        outputs_grad_inputs = self.outputs * (1.0 - self.outputs)
        return outputs_grads * outputs_grad_inputs


class ReLU(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        self.inputs = inputs
        return torch.relu(inputs)

    def backward(self, outputs_grads):
        return outputs_grads * (self.inputs > 0).float()


class MLP_3L(Op):
    def __init__(self, layers_size):
        self.fc1 = Linear(layers_size[0], layers_size[1], name='fc1')
        # ReLU激活函数
        self.act_fn1 = ReLU()
        self.fc2 = Linear(layers_size[1], layers_size[2], name='fc2')
        self.act_fn2 = ReLU()
        self.fc3 = Linear(layers_size[2], layers_size[3], name='fc3')
        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2, self.fc3]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        z3 = self.fc3(a2)
        return z3

    def backward(self, loss_grad_z3):
        loss_grad_a2 = self.fc3.backward(loss_grad_z3)
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        loss_grad_inputs = self.fc1.backward(loss_grad_z1)
        return loss_grad_inputs
