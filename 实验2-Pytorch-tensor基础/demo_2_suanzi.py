import torch
class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    # 前向函数
    # 输入：张量inputs
    # 输出：张量outputs
    def forward(self, inputs):
        # return outputs
        raise NotImplementedError

    # 反向函数
    # 输入：forward输出张量的梯度outputs_grads
    # 输出：forward输入张量的梯度inputs_grads
    def backward(self, outputs_grads):
        # return inputs_grads
        raise NotImplementedError


#==============加法的实现==============
class add(Op):
    def __init__(self):
        super(add, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        outputs = x + y
        return outputs

    def backward(self, grads):
        grads_x = grads * 1
        grads_y = grads * 1
        return grads_x, grads_y
#=============乘法的实现============
class multiply(Op):
    def __init__(self):
        super(multiply, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        outputs = x * y
        return outputs

    def backward(self, grads):
        grads_x = grads * self.y
        grads_y = grads * self.x
        return grads_x, grads_y
#=================指数算子================
import math
class exponential(Op):
    def __init__(self):
        super(exponential, self).__init__()

    def forward(self, x):
        self.x = x
        outputs = math.exp(x)
        return outputs

    def backward(self, grads):
        grads = grads * math.exp(self.x)
        return grads
a, b, c, d = 2, 3, 2, 2
# 实例化算子
multiply_op = multiply()
add_op = add()
exp_op = exponential()
y = exp_op(add_op(multiply_op(a, b), multiply_op(c, d)))
print('y: ', y)
#===================自动微分机制=====================
# 定义张量a，stop_gradient=False代表进行梯度传导
a = torch.tensor(2.0, requires_grad=True)
# 定义张量b，stop_gradient=True代表不进行梯度传导
b = torch.tensor(5.0, requires_grad=False)
c = a * b
# 自动计算反向梯度
print("Tensor a's grad is: {}".format(a.grad))
print("Tensor b's grad is: {}".format(b.grad))
print("Tensor c's grad is: {}".format(c.grad))