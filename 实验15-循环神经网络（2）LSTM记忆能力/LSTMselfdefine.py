import torch.nn.functional as F
import torch
import torch.nn as nn

# 声明LSTM和相关参数
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, Wi_attr=None, Wf_attr=None, Wo_attr=None, Wc_attr=None,
                 Ui_attr=None, Uf_attr=None, Uo_attr=None, Uc_attr=None, bi_attr=None, bf_attr=None,
                 bo_attr=None, bc_attr=None):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化模型参数
        if Wi_attr==None:
             Wi= torch.zeros(size=[input_size, hidden_size], dtype=torch.float32)
        else:
             Wi = torch.tensor(Wi_attr, dtype=torch.float32)
        self.W_i = torch.nn.Parameter(Wi)

        if Wf_attr==None:
             Wf=torch.zeros(size=[input_size, hidden_size], dtype=torch.float32)
        else:
             Wf = torch.tensor(Wf_attr, dtype=torch.float32)
        self.W_f = torch.nn.Parameter(Wf)

        if Wo_attr==None:
             Wo=torch.zeros(size=[input_size, hidden_size], dtype=torch.float32)
        else:
             Wo = torch.tensor(Wo_attr, dtype=torch.float32)
        self.W_o =torch.nn.Parameter(Wo)

        if Wc_attr==None:
            Wc=torch.zeros(size=[input_size, hidden_size], dtype=torch.float32)
        else:
            Wc = torch.tensor(Wc_attr, dtype=torch.float32)
        self.W_c = torch.nn.Parameter(Wc)

        if Ui_attr==None:
            Ui = torch.zeros(size=[hidden_size, hidden_size], dtype=torch.float32)
        else:
            Ui = torch.tensor(Ui_attr, dtype=torch.float32)
        self.U_i = torch.nn.Parameter(Ui)
        if Uf_attr == None:
            Uf = torch.zeros(size=[hidden_size, hidden_size], dtype=torch.float32)
        else:
            Uf = torch.tensor(Uf_attr, dtype=torch.float32)
        self.U_f = torch.nn.Parameter(Uf)

        if Uo_attr == None:
            Uo = torch.zeros(size=[hidden_size, hidden_size], dtype=torch.float32)
        else:
            Uo = torch.tensor(Uo_attr, dtype=torch.float32)
        self.U_o = torch.nn.Parameter(Uo)

        if Uc_attr == None:
            Uc = torch.zeros(size=[hidden_size, hidden_size], dtype=torch.float32)
        else:
            Uc = torch.tensor(Uc_attr, dtype=torch.float32)
        self.U_c = torch.nn.Parameter(Uc)

        if bi_attr == None:
            bi = torch.zeros(size=[1,hidden_size], dtype=torch.float32)
        else:
            bi = torch.tensor(bi_attr, dtype=torch.float32)
        self.b_i = torch.nn.Parameter(bi)
        if bf_attr == None:
            bf = torch.zeros(size=[1,hidden_size], dtype=torch.float32)
        else:
            bf = torch.tensor(bf_attr, dtype=torch.float32)
        self.b_f = torch.nn.Parameter(bf)

        if bo_attr == None:
            bo = torch.zeros(size=[1,hidden_size], dtype=torch.float32)
        else:
            bo = torch.tensor(bo_attr, dtype=torch.float32)
        self.b_o = torch.nn.Parameter(bo)
        if bc_attr == None:
            bc = torch.zeros(size=[1,hidden_size], dtype=torch.float32)
        else:
            bc = torch.tensor(bc_attr, dtype=torch.float32)
        self.b_c = torch.nn.Parameter(bc)

    # 初始化状态向量和隐状态向量
    def init_state(self, batch_size):
        hidden_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)
        cell_state = torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32)
        return hidden_state, cell_state

    # 定义前向计算
    def forward(self, inputs, states=None):
        # inputs: 输入数据，其shape为batch_size x seq_len x input_size
        batch_size, seq_len, input_size = inputs.shape

        # 初始化起始的单元状态和隐状态向量，其shape为batch_size x hidden_size
        if states is None:
            states = self.init_state(batch_size)
        hidden_state, cell_state = states

        # 执行LSTM计算，包括：输入门、遗忘门和输出门、候选内部状态、内部状态和隐状态向量
        for step in range(seq_len):
            # 获取当前时刻的输入数据step_input: 其shape为batch_size x input_size
            step_input = inputs[:, step, :]
            # 计算输入门, 遗忘门和输出门, 其shape为：batch_size x hidden_size
            I_gate = F.sigmoid(torch.matmul(step_input, self.W_i) + torch.matmul(hidden_state, self.U_i) + self.b_i)
            F_gate = F.sigmoid(torch.matmul(step_input, self.W_f) + torch.matmul(hidden_state, self.U_f) + self.b_f)
            O_gate = F.sigmoid(torch.matmul(step_input, self.W_o) + torch.matmul(hidden_state, self.U_o) + self.b_o)
            # 计算候选状态向量, 其shape为：batch_size x hidden_size
            C_tilde = F.tanh(torch.matmul(step_input, self.W_c) + torch.matmul(hidden_state, self.U_c) + self.b_c)
            # 计算单元状态向量, 其shape为：batch_size x hidden_size
            cell_state = F_gate * cell_state + I_gate * C_tilde
            # 计算隐状态向量，其shape为：batch_size x hidden_size
            hidden_state = O_gate * F.tanh(cell_state)

        return hidden_state
if __name__ == "__main__":
    torch.seed()
    # 这里创建一个随机数组作为测试数据，数据shape为batch_size x seq_len x input_size
    batch_size, seq_len, input_size, hidden_size = 2, 5, 10, 10
    inputs = torch.randn([batch_size, seq_len, input_size])
    # 设置模型的hidden_size
    torch_lstm = nn.LSTM(input_size, hidden_size, bias=True)
    # 获取torch_lstm中的参数，并设置相应的paramAttr,用于初始化lstm
    print(torch_lstm.weight_ih_l0.T.shape)
    chunked_W = torch.split(torch_lstm.weight_ih_l0.T, split_size_or_sections=10, dim=-1)
    chunked_U = torch.split(torch_lstm.weight_hh_l0.T, split_size_or_sections=10, dim=-1)
    chunked_b = torch.split(torch_lstm.bias_hh_l0.T, split_size_or_sections=10, dim=-1)

    Wi_attr = chunked_W[0]
    Wf_attr = chunked_W[1]
    Wc_attr = chunked_W[2]
    Wo_attr = chunked_W[3]
    Ui_attr = chunked_U[0]
    Uf_attr = chunked_U[1]
    Uc_attr = chunked_U[2]
    Uo_attr = chunked_U[3]
    bi_attr = chunked_b[0]
    bf_attr = chunked_b[1]
    bc_attr = chunked_b[2]
    bo_attr = chunked_b[3]
    self_lstm = LSTM(input_size, hidden_size, Wi_attr=Wi_attr, Wf_attr=Wf_attr, Wo_attr=Wo_attr, Wc_attr=Wc_attr,
                     Ui_attr=Ui_attr, Uf_attr=Uf_attr, Uo_attr=Uo_attr, Uc_attr=Uc_attr,
                     bi_attr=bi_attr, bf_attr=bf_attr, bo_attr=bo_attr, bc_attr=bc_attr)

    # 进行前向计算，获取隐状态向量，并打印展示
    self_hidden_state = self_lstm(inputs)
    torch_outputs, (torch_hidden_state, _) = torch_lstm(inputs)
    print("torch SRN:\n", torch_hidden_state.detach().numpy().squeeze(0))
    print("self SRN:\n", self_hidden_state.detach().numpy())

