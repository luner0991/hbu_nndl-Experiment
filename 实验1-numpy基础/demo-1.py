# 1. 导入numpy库
import numpy
import numpy as np
import  matplotlib.pyplot as plt

# 2. 建立一个一维数组 a 初始化为[4,5,6]
a = np.array([4,5,6])
print("a的类型:",type(a)) #  (1)输出a 的类型（type）
print("a的各维度大小：",a.shape)
print("a的第一个元素：",a[0])
# 3. 建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]]
b = np.array([[4,5,6],
              [1,2,3]])
print("b的各维度大小：",b.shape) # (1)输出各维度的大小（shape）
# (2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2)
print("b(0,0):",b[0,0])
print("b(0,1):",b[0,1])
print("b(1,1):",b[1,1])
# 4.
a = np.zeros((3,3),dtype= int)# (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）
b = np.ones((4,5))# (2)建立一个全1矩阵b,大小为4x5
c = np.eye(4)# (3)建立一个单位矩阵c ,大小为4x4
d = np.random.random(size=(3,2))# (4)生成一个随机数矩阵d,大小为 3x2
# 5. 建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] )
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print('数组a:', a)# (1)打印a;
print("a(2,3):", a[2, 3])# (2)输出 下标为(2,3),(0,0) 这两个数组元素的值
print("a(0,0):", a[0, 0])
# 6. 把上一题的 a数组的 0到1行 2到3列，放到b里面去
b = a[0:2,2:4]
print("数组b为",b)# (1),输出b;
print("b(0,0):",b[0,0])# (2) 输出b 的（0,0）这个元素的值
# 7. 把第5题中数组a的最后两行所有元素放到 c
c = a[1:3]
print("数组c:", c)# (1)输出 c ;
print("数组c中第一行的最后一个元素：", c[0, -1])# (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）
# 8. 建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素
a = np.array([[1,2],
              [3,4],
              [5,6]])
print("数组a：",a)
print("输出 （0,0）（1,1）（2,0）这三个元素：",a[[0, 1, 2], [0, 1, 0]])#高级索引
# 9. 建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0,0),(1,2),(2,0),(3,1)
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
print("数组a：",a)
b = np.array([0,2,0,1])
print("输出(0,0),(1,2),(2,0),(3,1)：",a[np.arange(4),b])
# 10. 对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a
a[np.arange(4), b] += 10
# 11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型
x = np.array([1,2])
print("x的数据类型：",type(x))
# 12. 执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
x = np.array([1.0,2.0])
print("x的数据类型：",type(x))
# 13. 执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x+y)
print(np.add(x,y))
# 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)
print(x-y,'\n',np.subtract(x,y))
# 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有 np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。
print(x*y)
print(np.multiply(x, y))
print(np.dot(x,y))
x_new = np.array([[1,2,3],[4,5,6]],dtype=np.float64)
y_new = np.array([[7,8],[10,11],[12,13]],dtype=np.float64)
# print(np.multiply(x_new,y_new))
print(np.dot(x_new,y_new))
# 16. 利用13题目中的x,y,输出 x / y
print(x/y,'\n',np.divide(x,y))
# 17. 利用13题目中的x,输出 x的 开方
print(x**0.5)
print(np.sqrt(x))
# 18. 利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))
print(x.dot(y))
print(np.dot(x,y))
# 19. 利用13题目中的 x,进行求和。
print(np.sum(x)) # (1)
print(np.sum(x,axis =0 )) # (2)
print(np.sum(x,axis = 1)) # (3)
# 20. 利用13题目中的 x,进行求平均数
print(np.mean(x))# (1)
print(np.mean(x,axis = 0))# (2)
print(np.mean(x,axis = 1))# (3)
# 21. 利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果
print("数组x:",x)
print("数组x的转置:",x.T)
# 22. 利用13题目中的x,求e的指数
print("e的指数:",np.exp(x))
# 23. 利用13题目中的 x,求值最大的下标
print(np.argmax(x))#(1)
print(np.argmax(x, axis =0))# (2)
print(np.argmax(x,axis =1))# (3)
# 24.画图，y=x*x 其中 x = np.arange(0, 100, 0.1)
x = np.arange(0,100,0.1)
plt.plot(x,x*x)
plt.show()
# 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)
x = np.arange(0,3 * np.pi, 0.1)
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.show()


'''
np.random.random((2, 3))  # 生成一个 2x3 的[0.0, 1.0)区间的随机数组
np.random.rand(2, 3)  # 生成一个 2x3 的[0.0, 1.0)区间的随机数组
np.random.randint(0, 10, size=(3, 3))  # 生成 0 到 10 之间的随机整数数组
np.random.randn(2, 3)  # 生成一个 2x3 的标准正态分布随机数组
'''
import numpy as np

a = np.array([1, 2, 3])
print(a.dtype)  # int32
