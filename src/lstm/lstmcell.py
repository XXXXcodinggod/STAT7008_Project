import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门参数
        self.W_ii = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        
        # 输出门参数
        self.W_io = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        # 候选单元状态参数
        self.W_ig = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        
        # 输出参数
        self.W_hq = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_q = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_t=None, c_t=None):
        """
        x:   (batch_size, seq_len, input_size)
        h_t: (batch_size, hidden_size)
        c_t: (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        # 初始化隐藏状态和细胞状态
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device).detach()
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device).detach()
            
        hidden_seq = []
        x_copy = x.transpose(1, 0)  # (seq_len, batch_size, input_size)
        
        for x_t in x_copy:
            # 输入门
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            
            # 遗忘门
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            
            # 输出门
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            
            # 候选单元状态
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
        
            # 更新单元状态
            c_t = f_t * c_t + i_t * g_t
            
            # 更新隐藏状态
            h_t = o_t * torch.tanh(c_t)
            
            # 记录当前隐藏状态
            hidden_seq.append(h_t.unsqueeze(1))
                        
        # 将所有时间步的隐藏状态堆叠成一个张量
        hidden_seq = torch.cat(hidden_seq, dim=1)  # (batch_size, seq_len, hidden_size)
        
        return hidden_seq, h_t, c_t

    