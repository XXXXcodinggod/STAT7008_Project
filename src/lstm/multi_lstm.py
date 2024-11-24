import torch
import torch.nn as nn
from .lstmcell import LSTMCell

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.LSTMCell = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size)])
        for _ in range(1, num_layers):
            self.LSTMCell.append(LSTMCell(self.hidden_size, self.hidden_size))

    def forward(self, input_seq, h_0=None, c_0=None):
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, 
                              input_seq.shape[0], 
                              self.hidden_size, 
                              device=input_seq.device).detach()
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, 
                              input_seq.shape[0], 
                              self.hidden_size, 
                              device=input_seq.device).detach()
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
    
    def forward(self, input_seq, h_0=None, c_0=None):
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, 
                              input_seq.shape[0], 
                              self.hidden_size, 
                              device=input_seq.device).detach() # (num_layer, batch_size, hidden_size)
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, 
                              input_seq.shape[0], 
                              self.hidden_size, 
                              device=input_seq.device).detach() # (num_layer, batch_size, hidden_size)
        output, _, _ = self.lstm(input_seq, h_0, c_0) # (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = output.shape
        output = output.view(batch_size * seq_len, self.hidden_size) # (batch_size * seq_len, hidden_size)
        output = self.fc(output)
        output = output.view(batch_size, seq_len, self.output_size)
        return output
    