'''
@Function: 使用numpy实现SRN
@Author: lxy
@Date: 2024/11/24
'''
import numpy as np
# 初始化输入序列
inputs = np.array([[1.,1.],[1.,1.],[2.,2.]])
print(f"inputs is:\n{inputs}")
# 初始化存储器
state_t = np.zeros(2)
print(f"states_t is:\n{state_t}")
# 初始化权重参数，这里所有权重都为1，bias = 0
w1, w2, w3, w4, w5, w6, w7, w8 = 1., 1., 1., 1., 1., 1., 1., 1.
U1, U2, U3, U4 = 1., 1., 1., 1.
print('============================================')
for t,input_t in enumerate(inputs):
    print(f"第{t+1}时刻：")
    print(f'input_t is: {input_t}')
    print(f'state_t is: {state_t}')
    # 隐藏层的输入=当前输入 + 前一时刻的状态
    input_h1 = np.dot([w1,w3],input_t) + np.dot([U2,U4],state_t)
    input_h2 = np.dot([w2,w4],input_t) + np.dot([U1,U3],state_t)
    # state_t = input_h1, input_h2 # 更新状态（无激活 输出=输入）：直接将计算得到的隐藏层输入赋值给state_t
    state_t = np.tanh(input_h1),np.tanh(input_h2) # 更新状态（有激活 输出=tanh(输入) ）
    # 输入为隐藏层的输出，计算最终输出
    output_y1 = np.dot([w5, w7], state_t)
    output_y2 = np.dot([w6, w8], state_t)
    print(f"outputs is: {output_y1}、{output_y2}")
    print('============================================')
