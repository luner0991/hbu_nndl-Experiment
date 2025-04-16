import torch
#========create torch-指定数据创建=========================
''''# 创建一维张量
ndim_1_Tensor = torch.tensor([2.0,3.0,4.0])
print(ndim_1_Tensor)
# 创建二维Tensor
ndim_2_Tensor = torch.tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(ndim_2_Tensor)
# 创建多维Tensor
ndim_3_Tensor = torch.tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(ndim_3_Tensor)
# 尝试定义在不同维度上元素数量不等的Tensor
# ndim_2_Tensor = torch.tensor([[1.0, 2.0],
#                                   [4.0, 5.0, 6.0]])
#========create torch-指定形状创建===================================
m,n=2,3
# 使用torch.zeros创建数据全为0，形状为[m, n]的Tensor
zeros_Tensor = torch.zeros([m,n])
# 使用torch.ones创建数据全为1，形状为[m, n]的Tensor
ones_Tensor = torch.ones([m,n])
# 使用torch.full创建数据全为指定值，形状为[m, n]的Tensor，这里我们指定数据为10
full_Tensor = torch.full([m,n],10)

print("zeros_Tensor:",zeros_Tensor)
print("ones_Tensor:",ones_Tensor)
print("full_Tensor:",full_Tensor)
#========create torch-指定区间创建=====================================
# 使用torch.arange创建以步长step均匀分隔数值区间[start, end)的一维Tensor
arange_Tensor = torch.arange(start=1, end=5, step=1)
# 使用torch.linspace创建以元素个数num均匀分隔数值区间[start, end]的Tensor
linspace_Tensor = torch.linspace(start=1, end=5, steps=9)

print('arange Tensor: ', arange_Tensor)
print('linspace Tensor: ', linspace_Tensor)
#=================张量的形状==============================
ndim_4_Tensor = torch.ones([2, 3, 4, 5])
print("Number of dimensions:", ndim_4_Tensor.ndim)
print("Shape of Tensor:", ndim_4_Tensor.shape)
print("Elements number along axis 0 of Tensor:", ndim_4_Tensor.shape[0])
print("Elements number along the last axis of Tensor:", ndim_4_Tensor.shape[-1])
print('Number of elements in Tensor: ', ndim_4_Tensor.size)
#=================================改变形状属性->reshape===========================
#定义一个shape为[3,2,5]的三维Tensor
ndim_3_Tensor = torch.tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]],
                                  [[21, 22, 23, 24, 25],
                                   [26, 27, 28, 29, 30]]])
print("the shape of ndim_3_Tensor:", ndim_3_Tensor.shape)

# torch.reshape 可以保持在输入数据不变的情况下，改变数据形状。这里我们设置reshape为[2,5,3]
reshape_Tensor = torch.reshape(ndim_3_Tensor, [2, 5, 3])
print("After reshape:\n", reshape_Tensor)
#==========================改变形状属性->view=============================
new_Tensor1_pytorch = ndim_3_Tensor.view([-1])
print('new Tensor 1 shape: ', new_Tensor1_pytorch.shape)
print(new_Tensor1_pytorch)
new_Tensor2_pytorch = ndim_3_Tensor.view([3, 5, 2])
print('new Tensor 2 shape: ', new_Tensor2_pytorch.shape)
#=================================张量的数据类型========================
# 使用torch.tensor通过已知数据来创建一个Tensor
print("Tensor dtype from Python integers:", torch.tensor(1).dtype)
print("Tensor dtype from Python floating point:", torch.tensor(1.0).dtype)
#=============================改变张量的数据类型==================================
# 定义dtype为float32的Tensor
float32_Tensor = torch.tensor(1.0)
# Tensor.type可以将输入数据的数据类型转换为指定的dtype并输出。支持输出和输入数据类型相同。
#int64_Tensor = float32_Tensor.type(torch.int64)
int64_Tensor = float32_Tensor.to(torch.int64)
print("Tensor after type to int64:", int64_Tensor.dtype)
#===========张量的设备位置=========
# 创建CPU上的PyTorch Tensor
cpu_Tensor = torch.tensor(1)
# 通过Tensor.device查看张量所在设备位置
print('CPU Tensor: ', cpu_Tensor.device)
# 创建GPU上的PyTorch Tensor
if torch.cuda.is_available():
    gpu_Tensor = torch.tensor([2,3,4,5], device=torch.device('cuda:0'))#指定了设备，cuda:0表示第一个GPU
    # gpu_Tensor = torch.tensor([2,3,4,5], device='cuda')# 允许PyTorch自动选择一个可用的CUDA设备
    print('GPU Tensor: ', gpu_Tensor.device)
else:
    print('GPU is not available.')
#===============numpy数组与torch的==============
ndim_1_Tensor = torch.tensor([1., 2.])
# 将当前 Tensor 转化为 numpy.ndarray
print('Tensor to numpy: ', ndim_1_Tensor.numpy())
#========================通过索引/切片访问张量========
# 定义1个一维Tensor
ndim_1_Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

print("原型张量:", ndim_1_Tensor)
print("第一个元素:", ndim_1_Tensor[0])
print("最后一个元素:", ndim_1_Tensor[-1])
print("所有元素:", ndim_1_Tensor[:])
print("前3个元素:", ndim_1_Tensor[:3])
print("步长为3:", ndim_1_Tensor[::3])
# print("Reverse:", ndim_1_Tensor[::-1])->Paddle支持使用Tensor[::-1]的方式进行Reverse，Pytorch不支持此功能
#=======================张量的修改====================
# 定义1个二维Tensor
ndim_2_Tensor = torch.ones([2, 3], dtype=torch.float32)
print('原始张量: ', ndim_2_Tensor)
ndim_2_Tensor[0] = 0
print('修改第一维为0: ', ndim_2_Tensor)
ndim_2_Tensor[0:1] = 2.1
print('修改第1维为2.1: ', ndim_2_Tensor)
ndim_2_Tensor[...] = 3
print('修改全部Tensor: ', ndim_2_Tensor)
#=================张量的运算==============
# 定义两个Tensor
x = torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float64)
y = torch.tensor([[5.5, 6.6], [7.7, 8.8]], dtype=torch.float64)
# 第一种调用方法，paddle.add逐元素相加算子，并将各个位置的输出元素保存到返回结果中
print('Method 1: ', torch.add(x, y))
# 第二种调用方法
print('Method 2: ', x.add(y))

# 张量的数学运算
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([-1,-2,-3,-4,-5])
print("x:",x)
print("y:",y)
print("y.abs():",y.abs())
print("x.exp():",x.exp())
print("x.log():",x.log())
print("x.reciprocal():",x.reciprocal())
print("x.square():",x.square())
print("x.sqrt():",x.sqrt())
print("x.sin():",x.sin())
print("x.cos():",x.cos())
print("x.subtract(y):",x.subtract(y))
print("x.divide(y):",x.divide(y))
print("x.pow(y):",x.pow(y))
print("x.max():",x.max())
print("x.min():",x.min())
print("x.prod():",x.prod())
print("x.sum():",x.sum())
#=============# 张量的逻辑运算===========
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,3,3,4,4])
print("x:",x)
print("y:",y)
print("x.isfinite():",x.isfinite())
# print("x.equal_all(y):",x.equal_all(y))
print("x.equal(y):",x.equal(y))
print("x.not_equal(y):",x.not_equal(y))
# print("x.less_than(y):",x.less_than(y))
print("x.less_equal(y):",x.less_equal(y))
# print("x.greater_than(y):",x.greater_than(y))
print("x.greater_equal(y):",x.greater_equal(y))
print("x.allclose(y):",x.allclose(y))
#================张量的矩阵运算=============
x = torch.tensor([[1,2,3],
            [4,5,6]],dtype=torch.float32)
y = torch.tensor([[1,1,1],
            [2,2,2]],dtype=torch.float32)
print("x:",x)
print("y:",y)
print("x.t():",x.t())
# print("x.transpose([1,0]):",x.transpose([1,0]))
print("x.norm('fro'):",x.norm('fro'))
print("x.dist(y,p=2):",x.dist(y,p=2))
print("x.matmul(y):",x.matmul(y.t()))
#===============广播机制======================
# 当两个Tensor的形状一致时，可以广播
x = torch.ones((2, 3, 4))
y = torch.ones((2, 3, 4))
z = x + y
print('两个形状一样的张量广播 ', z.shape)

x = torch.ones((2, 3, 1, 5))
y = torch.ones((3, 4, 1))
# 从后往前依次比较：
# 第一次：y的维度大小是1
# 第二次：x的维度大小是1
# 第三次：x和y的维度大小相等，都为3
# 第四次：y的维度不存在
# 所以x和y是可以广播的
z = x + y
print('两个形状不一样的张量广播:', z.shape)
#======不能进行广播=========
x = torch.ones((2, 3, 4))
y = torch.ones((2, 3, 6))
z = x + y'''
#=============乘法的广播=========
x = torch.ones([10, 1, 5, 2])
y = torch.ones([3, 2, 5])
z = torch.matmul(x, y)
print('After matmul: ', z.shape)