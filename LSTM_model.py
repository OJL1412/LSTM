import torch
import torch.nn as nn


class M_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_state=None, cell_state=None, drop_out=0.1, bind_emb=True):
        super(M_LSTM, self).__init__()

        self.v_size = vocab_size

        hidden_state = emb_dim if hidden_state is None else hidden_state  # 给定一个默认的输出值
        cell_state = emb_dim if cell_state is None else cell_state  # 给定一个默认的输出值

        # Embedding进行词嵌入，随机初始化映射为一个向量矩阵，参数1是嵌入字典的词的数量，参数2是每个嵌入向量的大小，此处为词向量维度32
        self.w_emb = nn.Embedding(self.v_size, emb_dim)

        self.i_gate = nn.Sequential(
            nn.Linear(emb_dim + hidden_state, hidden_state, bias=False),
            nn.LayerNorm(hidden_state),
            nn.Sigmoid(),
            nn.Dropout(drop_out)
        )

        self.f_gate = nn.Sequential(
            nn.Linear(emb_dim + hidden_state, hidden_state, bias=False),
            nn.LayerNorm(hidden_state),
            nn.Sigmoid(),
            nn.Dropout(drop_out)
        )

        self.o_gate = nn.Sequential(
            nn.Linear(emb_dim + hidden_state, hidden_state, bias=False),
            nn.LayerNorm(hidden_state),
            nn.Sigmoid(),
            nn.Dropout(drop_out)
        )

        self.act = nn.Sequential(
            nn.Linear(emb_dim + hidden_state, hidden_state, bias=False),
            nn.LayerNorm(hidden_state),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_state, emb_dim, bias=False),
            nn.Linear(emb_dim, self.v_size)  # 32 --> vocab_size
        )

        self.hidden_state_init = nn.Parameter(torch.zeros(1, hidden_state))  # 隐状态初始化

        if bind_emb:
            self.classifier[-1].weight = self.w_emb.weight  # 绑定词向量与分类器的权重

    def forward(self, input):
        output = self.hidden_state_init.expand(input.size(0), -1)
        cell = self.hidden_state_init.expand(input.size(0), -1)
        lstm_input = self.w_emb(input)

        _state = []
        for i in lstm_input.unbind(1):
            concat_result = torch.cat((i, output), dim=-1)

            f_gate_state = self.f_gate(concat_result)
            o_gate_state = self.o_gate(concat_result)
            i_gate_state = self.i_gate(concat_result)
            h_state = self.act(concat_result)

            product_cell_fgate = cell * f_gate_state
            product_igate_h = i_gate_state * h_state

            current_cell = product_cell_fgate + product_igate_h
            current_output = current_cell * o_gate_state

            output = current_output
            cell = current_cell

            _state.append(output)

        state = self.classifier(torch.stack(_state, dim=1))

        return state

    def decode(self, input):
        output_state = self.hidden_state_init.expand(input.size(0), -1)
        cell_state = self.hidden_state_init.expand(input.size(0), -1)
        lstm_input = self.w_emb(input)

        concat_result = torch.cat((lstm_input, output_state), dim=-1)

        f_gate_state = self.f_gate(concat_result)
        o_gate_state = self.o_gate(concat_result)
        i_gate_state = self.i_gate(concat_result)
        h_state = self.act(concat_result)

        product_cell_fgate = cell_state * f_gate_state
        product_igate_h = i_gate_state * h_state

        current_cell = product_cell_fgate + product_igate_h
        current_output = current_cell * o_gate_state

        output = torch.argmax(self.classifier(current_output), dim=1)

        return output
