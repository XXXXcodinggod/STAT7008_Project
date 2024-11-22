import torch
import torch.nn as nn
from src.lstm.lstmcell import LSTMCell

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.LSTMCell = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size)])
        for _ in range(1, num_layers):
            self.LSTMCell.append(LSTMCell(self.hidden_size, self.hidden_size))

    def forward(self, input_seq, h_0, c_0):
        hidden_seq, h, c = self.LSTMCell[0](input_seq, h_0[0], c_0[0])
        for i in range(1,self.num_layers):
            hidden_seq, h, c = self.LSTMCell[i](hidden_seq, h_0[i], c_0[i])
        return hidden_seq, h, c

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = MultiLayerLSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers, input_seq.shape[0], self.hidden_size).detach()
        c_0 = torch.zeros(self.num_layers, input_seq.shape[0], self.hidden_size).detach()
        output, _, _ = self.lstm(input_seq, h_0, c_0) # (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = output.shape
        output = output.view(batch_size * seq_len, self.hidden_size) # (batch_size * seq_len, hidden_size)
        output = self.fc(output)
        output = output.view(batch_size, seq_len, self.output_size)
        return output
        
if __name__ == '__main__':
    batch_size = 64
    n_layers = 4
    x = torch.randn(batch_size, 20, 10)
    lstm = MultiLayerLSTM(10, 32, n_layers)
    h_0 = torch.randn(n_layers, batch_size, 32)
    c_0 = torch.randn(n_layers, batch_size, 32)
    output_seq, h, c = lstm(x, h_0, c_0)

    print('hidden_seq.shape:', output_seq.shape) # (batch_size, seq_len, hidden_size)
    print('h.shape:', h_0.shape) # (batch_size, hidden_size)
    print('c.shape:', c_0.shape) # (batch_size, hidden_size)
    
    output_size = 24
    lstm = LSTM(10, 32, output_size, n_layers)
    output_seq = lstm(x)
    print('output.shape:', output_seq.shape) # (batch_size, seq_len, output_size)