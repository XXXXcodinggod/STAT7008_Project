import torch
import torch.nn as nn
from src.lstm.lstmcell import LSTMCell

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTMCell_forward = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size)])
        self.LSTMCell_Backward = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size)])
        for _ in range(1, num_layers):
            self.LSTMCell_forward.append(LSTMCell(2 * self.hidden_size, self.hidden_size))
            self.LSTMCell_Backward.append(LSTMCell(2 * self.hidden_size, self.hidden_size))
        
    def forward(self, input_seq, h_0, c_0):
        input_seq_reversed = input_seq.flip(dims=[1]) # (batch_size, seq_len, hidden_size)
        
        hidden_seq_forward, h_forward, c_forward = self.LSTMCell_forward[0](input_seq, h_0[0], c_0[0]) # (batch_size, seq_len, hidden_size)
        hidden_seq_backward, h_backward, c_backward = self.LSTMCell_Backward[0](input_seq_reversed, h_0[self.num_layers], c_0[self.num_layers]) # (batch_size, seq_len, hidden_size)
        hidden_seq_combined = torch.cat((hidden_seq_forward, hidden_seq_backward.flip(dims=[1])), dim=2) # (batch_size, seq_len, 2 * hidden_size)
        
        for i in range(1,self.num_layers):
            hidden_seq_forward, h_forward, c_forward = self.LSTMCell_forward[i](hidden_seq_combined, h_0[i], c_0[i])
            hidden_seq_backward, h_backward, c_backward = self.LSTMCell_Backward[i](hidden_seq_combined.flip(dims=[1]), h_0[self.num_layers + i], c_0[i])
            hidden_seq_combined = torch.cat((hidden_seq_forward, hidden_seq_backward.flip(dims=[1])), dim=2)
            
        return hidden_seq_combined, h_forward, h_backward, c_forward, c_backward

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = BidirectionalLSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(2 * self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        h_0 = torch.zeros(2 * self.num_layers, input_seq.shape[0], self.hidden_size).detach()
        c_0 = torch.zeros(2 * self.num_layers, input_seq.shape[0], self.hidden_size).detach()
        output, _, _, _, _ = self.lstm(input_seq, h_0, c_0) # (batch_size, seq_len, 2 * hidden_size)
        batch_size, seq_len, _ = output.shape
        output = output.view(batch_size * seq_len, 2 * self.hidden_size) # (batch_size * seq_len, 2 * hidden_size)
        output = self.fc(output)
        output = output.view(batch_size, seq_len, self.output_size)
        return output

if __name__ == '__main__':
    batch_size = 64
    n_layers = 4
    x = torch.randn(batch_size, 20, 10)
    lstm = BidirectionalLSTM(10, 32, n_layers)
    h_0 = torch.randn(2 * n_layers, batch_size, 32)
    c_0 = torch.randn(2 * n_layers, batch_size, 32)
    hidden_seq_combined, h_forward, h_backward, c_forward, c_backward = lstm(x, h_0, c_0)

    print('hidden_seq.shape:', hidden_seq_combined.shape) # (batch_size, seq_len, hidden_size)
    print('h_forward.shape:', h_forward.shape) # (batch_size, hidden_size)
    print('c_forward.shape:', c_forward.shape) # (batch_size, hidden_size)

    output_size = 24
    lstm = BiLSTM(10, 32, output_size, n_layers)
    output_seq = lstm(x)
    print('output.shape:', output_seq.shape) # (batch_size, seq_len, output_size)