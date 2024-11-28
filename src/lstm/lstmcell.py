import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """
    LSTMCell implements LSTM cell
    
    Method:
        forward(self, x, h_t, c_t): allow whole sequence
    """
    
    def __init__(self, input_size, hidden_size, dropout_rate=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # input gate parameters
        self.W_ii = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        nn.init.kaiming_uniform_(self.W_ii) 
        nn.init.kaiming_uniform_(self.W_hi) 
        
        # forget gate parameters
        self.W_if = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        nn.init.kaiming_uniform_(self.W_if) 
        nn.init.kaiming_uniform_(self.W_hf) 
        
        # output gate parameters
        self.W_io = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        nn.init.kaiming_uniform_(self.W_io) 
        nn.init.kaiming_uniform_(self.W_ho) 
        
        # candidate cell state parameters
        self.W_ig = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        nn.init.kaiming_uniform_(self.W_ig) 
        nn.init.kaiming_uniform_(self.W_hg) 

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, h_t=None, c_t=None):
        """
        x:   (batch_size, seq_len, input_size)
        h_t: (batch_size, hidden_size)
        c_t: (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        # update hidden state (batch_size, hidden_size)
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device).detach()
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device).detach()
            
        hidden_seq = []
        x_copy = x.transpose(1, 0)  # (seq_len, batch_size, input_size)
        
        for x_t in x_copy:
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i) # input gate
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f) # forget gate
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o) # output gate
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g) # candidate cell state
            c_t = f_t * c_t + i_t * g_t # update cell state (batch_size, hidden_size)
            h_t = o_t * torch.tanh(c_t) # update hidden state (batch_size, hidden_size)
            h_t = self.dropout(h_t)
            hidden_seq.append(h_t.unsqueeze(1)) # (batch_size, 1, hidden_size)
            
        hidden_seq = torch.cat(hidden_seq, dim=1)  # (batch_size, seq_len, hidden_size)
        
        return hidden_seq, h_t, c_t

class LSTMCell_(LSTMCell):
    """
    LSTMCell_ inherits from LSTMCell and is used for implementing teacher forcing
    
    Method:
        forward(self, x, h_t, c_t): only allow 1 timestamp 
    """
    
    def __init__(self, input_size, hidden_size, dropout_rate=0.0):
        super().__init__(input_size, hidden_size, dropout_rate)

    def forward(self, x, h_t=None, c_t=None):
        """
        x:   (batch_size, input_size)
        h_t: (batch_size, hidden_size)
        c_t: (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        # initialize hidden state and cell state
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device).detach()
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device).detach()
        
        i_t = torch.sigmoid(x @ self.W_ii + h_t @ self.W_hi + self.b_i) # input gate
        f_t = torch.sigmoid(x @ self.W_if + h_t @ self.W_hf + self.b_f) # forget gate
        o_t = torch.sigmoid(x @ self.W_io + h_t @ self.W_ho + self.b_o) # output gate
        g_t = torch.tanh(x @ self.W_ig + h_t @ self.W_hg + self.b_g) # candidate cell state
        c_t = f_t * c_t + i_t * g_t # update cell state (batch_size, hidden_size)
        h_t = o_t * torch.tanh(c_t) # update hidden state (batch_size, hidden_size)
        h_t = self.dropout(h_t)
        
        return h_t, c_t
