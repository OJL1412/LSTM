import torch
import torch.nn as nn


class LSTMCell_X(nn.Module):
    """
    x: 决定使用多少层LSTM
    i_size: 前馈神经网络的输入大小
    o_size: 即隐状态
    激活函数前加nn.LayerNorm: 先算o_size个数的均值和方差，再计算（o_size-均值）/方差，用以排除极大和极小的数，稳定数据
    激活函数后接nn.Dropout: 用来随机丢掉部分数据特征（弄成0），使得剩下的数据特征包含丢掉的数据的特征，使数据的特征间产生关联
    """

    def __init__(self, x, i_size, o_size=None, dropout=0.1):
        super(LSTMCell_X, self).__init__()

        o_size = i_size if o_size is None else o_size

        self.trans = nn.Linear(i_size + o_size, o_size * x, bias=False)
        self.norm = nn.LayerNorm([x, o_size])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, input, state):
        output, cell = state
        o_size = list(output.size())
        o_size.insert(-1, 4)

        # 第1步计算: Cell(i-1)和Output(i-1) --> Cell(i)和Output(i)
        handle = self.norm(self.trans(torch.cat((input, output), dim=-1)).view(o_size))

        f_gate, i_gate, o_gate = handle.narrow(-2, 0, 3).sigmoid().unbind(-2)
        h_state = self.drop(self.act(handle.select(-2, 3)))

        # 第2步计算
        cell = f_gate * cell + i_gate * h_state
        output = o_gate * cell

        return output, cell


class LSTMLayer(nn.Module):
    def __init__(self, i_size=32, o_size=None):
        super(LSTMLayer, self).__init__()

        o_size = i_size if o_size is None else o_size

        self.net = LSTMCell_X(4, i_size, o_size)

        self.init_hx = nn.Parameter(torch.zeros(1, o_size))
        self.init_cx = nn.Parameter(torch.zeros(1, o_size))

    def forward(self, input):
        hidden_state = self.init_hx.expand(input.size(0), -1)
        cell = self.init_cx.expand(input.size(0), -1)

        _state = []

        for i in input.unbind(1):
            hidden_state, cell = self.LC_net(i, (hidden_state, cell))
            _state.append(hidden_state)

        output = torch.stack(_state, dim=1)

        return output

    def decode(self, input, hidden_state=None, cell=None):
        hidden_state = self.init_hx.expand(input.size(0), -1) if hidden_state is None else hidden_state
        cell = self.init_cx.expand(input.size(0), -1) if cell is None else cell

        hidden_state, cell = self.net(input, (hidden_state, cell))

        return hidden_state, cell


class M_LSTM(nn.Module):
    def __init__(self, v_size, emb_size=32, h_size=None, bind_emb=True):
        super(M_LSTM, self).__init__()

        h_size = emb_size if h_size is None else h_size

        # Embedding进行词嵌入，随机初始化映射为一个向量矩阵，参数1是嵌入字典的词的数量，参数2是每个嵌入向量的大小，此处为词向量维度32
        self.w_emb = nn.Embedding(v_size, emb_size)
        self.net = LSTMLayer()

        self.classifier = nn.Sequential(
            nn.Linear(h_size, emb_size, bias=False),
            nn.Linear(emb_size, v_size)
        )

        if bind_emb:
            self.classifier[-1].weight = self.w_emb.weight

    def forward(self, input):
        lstm_input = self.w_emb(input)

        output = self.classifier(self.net(lstm_input))

        return output

    def decode(self, input, hidden_state, cell):
        lstm_input = self.w_emb(input)

        hidden_state, cell = self.net.decode(lstm_input, hidden_state, cell)

        output = torch.argmax(self.classifier(hidden_state), dim=1)

        return output, hidden_state, cell
