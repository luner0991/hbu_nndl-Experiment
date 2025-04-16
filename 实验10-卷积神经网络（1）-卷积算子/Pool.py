'''
@Author: lxy
@Functon: Implement pool2D Operator
@date: 2024/11/7
'''
import torch
import torch.nn as nn
class Pool2D(nn.Module):
    def __init__(self, size=(2, 2), mode='max', stride=1):
        super(Pool2D, self).__init__()
        # 汇聚方式
        self.mode = mode
        self.h, self.w = size
        self.stride = stride
    def forward(self, x):
        output_h = (x.shape[2] - self.h) // self.stride + 1
        output_w = (x.shape[3] - self.w) // self.stride + 1
        output = torch.zeros([x.shape[0], x.shape[1], output_h, output_w], device=x.device)

        # 汇聚
        for i in range(output_h):
            for j in range(output_w):
                # 最大汇聚
                if self.mode == 'max':
                    output[:, :, i, j] = torch.max(
                        x[:, :, self.stride*i:self.stride*i+self.h, self.stride*j:self.stride*j+self.w],
                        dim=2, keepdim=False)[0].max(dim=2)[0]
                # 平均汇聚
                elif self.mode == 'avg':
                    output[:, :, i, j] = torch.mean(
                        x[:, :, self.stride*i:self.stride*i+self.h, self.stride*j:self.stride*j+self.w],
                        dim=[2,3])
        return output

# 测试自定义汇聚层
inputs = torch.tensor([[[[1., 2., 3., 4.],
                         [5., 6., 7., 8.],
                         [9., 10., 11., 12.],
                         [13., 14., 15., 16.]]]], dtype=torch.float32)

pool2d = Pool2D(stride=2)
outputs = pool2d(inputs)
print("input: {}, \noutput: {}".format(inputs.shape, outputs.shape))

# 比较Maxpool2D与PyTorch API运算结果
maxpool2d_pytorch = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
outputs_pytorch = maxpool2d_pytorch(inputs)
print('Maxpool2D outputs:', outputs)
print('nn.MaxPool2d outputs:', outputs_pytorch)

# 比较Avgpool2D与PyTorch API运算结果
avgpool2d_pytorch = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
outputs_pytorch_avg = avgpool2d_pytorch(inputs)
pool2d_avg = Pool2D(mode='avg', stride=2)
outputs_avg = pool2d_avg(inputs)
print('Avgpool2D outputs:', outputs_avg)
print('nn.AvgPool2d outputs:', outputs_pytorch_avg)
