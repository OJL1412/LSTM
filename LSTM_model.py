import torch
import torch.nn as nn

"""
三部分: 
① 完整的语言模型M_LSTM: embedding, classifier, 封装的LSTMLayer类
② LSTMLayer类: 给一句话的向量表示，生成这句话新的向量表示（从前到后走一遍来处理一句话），该类不停调用LSTMCell进行计算
③ LSTMCell类: 每一步的output及cell的计算封装
"""


class LSTMCell(nn.Module):
    """
    <LSTMCell类: 将LSTM计算代码封装,相关参数说明如下>
    ① i_size: 前馈神经网络的输入大小
    ② h_state: 即隐状态
    ③ 激活函数前加nn.LayerNorm: 先算h_state个数的均值和方差，再计算（h_state-均值）/方差，用以排除极大和极小的数，稳定数据
    ④ 激活函数后接nn.Dropout: 用来随机丢掉部分数据特征（弄成0），使得剩下的数据特征包含丢掉的数据的特征，使数据的特征间产生关联
    ⑤ 关于3个门：
        1）遗忘门f_gate：控制上一个时刻的记忆单元cell(t−1)需要遗忘多少信息
        2）输入门i_gate：控制当前时刻的候选状态~cell(t)有多少信息需要存储
        3）输出门o_gate：控制当前时刻的记忆单元cell(t)有多少信息需要输出给外部状态h(t)
    """

    def __init__(self, i_size, h_state, dropout=None):
        super(LSTMCell, self).__init__()

        h_state = i_size if h_state is None else h_state
        dropout = 0.1 if dropout is None else dropout

        self.trans = nn.Linear(i_size + h_state, h_state * 4, bias=False)  # ?+?-->?*4
        self.norm = nn.LayerNorm((4, h_state))  # 归一化处理，得到4个？大小的张量
        self.act = nn.GELU()  # 激活函数处理
        self.drop = nn.Dropout(dropout)  # 使用dropout使数据特征间产生关联

    def forward(self, input, h_state):
        """
        -->Cell(i-1)和Output(i-1) --> Cell(i)和Output(i)的3步处理
        ① 拼接当前词向量input和上一个词的隐状态output（state），并转化为o_size形状的tensor
        ② 将第①步中得到的tensor经过3个门的计算，确定遗忘多少信息，存储多少信息，输出给外部h(t)多少信息
        ③ 相乘加和运算，更新output和cell
        """
        output, cell = h_state
        o_size = list(output.size())
        o_size.insert(-1, 4)  # 在最后一维插入4，[b_size, o_size]-->[b_size, 4, o_size]，然后将其传给遗忘门f_gate、输入门i_gate、输出门o_gate

        # 第1步处理
        handle = self.norm(self.trans(torch.cat((input, output), dim=-1)).view(o_size))

        # 第2步处理
        f_gate, i_gate, o_gate = handle.narrow(-2, 0, 3).sigmoid().unbind(-2)  # tensor前3行经sigmoid函数处理再按行解绑分别赋予3个门
        h_state = self.drop(self.act(handle.select(-2, 3)))  # 取tensor最后一行做act处理

        # 第3步处理
        cell = f_gate * cell + i_gate * h_state
        output = o_gate * cell

        return output, cell


class LSTMLayer(nn.Module):
    """
    <LSTMLayer类>
    ① 从前到后走一遍来处理一句话，该类不停调用LSTMCell进行计算
    """

    def __init__(self, i_size, h_state):
        super(LSTMLayer, self).__init__()

        self.l_net = LSTMCell(i_size, h_state)

        # hidden_state和cell初始化
        self.init_ht = nn.Parameter(torch.zeros(1, h_state))
        self.init_cl = nn.Parameter(torch.zeros(1, h_state))

    def forward(self, input):
        h_state = self.init_ht.expand(input.size(0), -1)
        cell = self.init_cl.expand(input.size(0), -1)

        _state = []

        for i in input.unbind(1):
            h_state, cell = self.l_net(i, (h_state, cell))
            _state.append(h_state)

        output = torch.stack(_state, dim=1)

        return output

    def decode(self, input, h_state=None):
        h_state, cell = self.init_ht.expand(input.size(0), -1), self.init_cl.expand(input.size(0), -1) if h_state is None else h_state

        # h_state = self.init_ht.expand(input.size(0), -1) if h_state is None else h_state
        # cell = self.init_cl.expand(input.size(0), -1) if cell is None else cell

        h_state, cell = self.l_net(input, (h_state, cell))

        return h_state, (h_state, cell)


class M_LSTM(nn.Module):
    """
    <M_LSTM类>
    """

    def __init__(self, v_size, num_layers, emb_dim=32, h_state=None, bind_emb=True):
        super(M_LSTM, self).__init__()

        h_state = emb_dim if h_state is None else h_state

        # Embedding进行词嵌入，随机初始化映射为一个向量矩阵，参数1是嵌入字典的词的数量，参数2是每个嵌入向量的大小，此处为词向量维度32
        self.w_emb = nn.Embedding(v_size, emb_dim)
        self.m_net = nn.Sequential(
            *[LSTMLayer(i_size=emb_dim if i == 0 else h_state, h_state=h_state) for i in range(num_layers-1)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(h_state, emb_dim, bias=False),
            nn.Linear(emb_dim, v_size)
        )

        if bind_emb:
            self.classifier[-1].weight = self.w_emb.weight

    def forward(self, input):
        lstm_input = self.w_emb(input)

        output = self.classifier(self.m_net(lstm_input))

        return output

    """
    ① input: 输入的语料，一般是一个词
    ② step: 需预测的词数
    """

    def decode(self, input, ht_record):
        inp = self.w_emb(input)  # (b_size, seql, v_size)
        rs = []

        # ht_record = {}  # 用来记录每次LSTMCell层计算完后的(h_state, cell)
        for i in range(len(self.m_net)):
            ht_record[i] = None

        for j, net in enumerate(self.m_net):
            inp, ht_record[j] = net.decode(inp, h_state=ht_record[j])

        output = torch.argmax(self.classifier(inp), dim=-1)

        return output, ht_record

