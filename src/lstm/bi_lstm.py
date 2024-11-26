import torch
import torch.nn as nn
from .lstmcell import LSTMCell

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
        
    def forward(self, input_seq, h_0=None, c_0=None):
        batch_size = input_seq.size(0)
        
        if h_0 is None:
            h_0 = torch.zeros(2 * self.num_layers, 
                              batch_size, 
                              self.hidden_size, 
                              device=input_seq.device).detach()
        if c_0 is None:
            c_0 = torch.zeros(2 * self.num_layers, 
                              batch_size, 
                              self.hidden_size, 
                              device=input_seq.device).detach()
        
        h_t_copy = torch.zeros(self.num_layers, 
                               batch_size, 
                               2 * self.hidden_size, 
                               device=input_seq.device)

        c_t_copy = torch.zeros(self.num_layers, 
                               batch_size, 
                               2 * self.hidden_size, 
                               device=input_seq.device)
        
        input_seq_reversed = input_seq.flip(dims=[1]) # (batch_size, seq_len, hidden_size)

        # hidden_seq_forward： (batch_size, seq_len, hidden_size)； h_forward, c_forward: (batch_size, hidden_size)
        hidden_seq_forward, h_forward, c_forward = self.LSTMCell_forward[0](input_seq, h_0[0], c_0[0]) # (batch_size, seq_len, hidden_size)
        hidden_seq_backward, h_backward, c_backward = self.LSTMCell_Backward[0](input_seq_reversed, h_0[self.num_layers], c_0[self.num_layers]) # (batch_size, seq_len, hidden_size)
        hidden_seq_combined = torch.cat((hidden_seq_forward, hidden_seq_backward.flip(dims=[1])), dim=2) # (batch_size, seq_len, 2 * hidden_size)
        h_combined, c_combined = torch.cat([h_forward, h_backward], dim=1), torch.cat([c_forward, c_backward], dim=1)
        h_t_copy[0], c_t_copy[0] = h_combined, c_combined
        
        for i in range(1,self.num_layers):
            hidden_seq_forward, h_forward, c_forward = self.LSTMCell_forward[i](hidden_seq_combined, h_0[i], c_0[i])
            hidden_seq_backward, h_backward, c_backward = self.LSTMCell_Backward[i](hidden_seq_combined.flip(dims=[1]), h_0[self.num_layers + i], c_0[self.num_layers + i])
            hidden_seq_combined = torch.cat((hidden_seq_forward, hidden_seq_backward.flip(dims=[1])), dim=2)
            h_combined, c_combined = torch.cat([h_forward, h_backward], dim=1), torch.cat([c_forward, c_backward], dim=1)
            h_t_copy[i], c_t_copy[i] = h_combined, c_combined

        return hidden_seq_combined, h_t_copy, c_t_copy

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = BidirectionalLSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(2 * self.hidden_size, self.output_size)
    
    def forward(self, input_seq, h_0=None, c_0=None):
        if h_0 is None:
            h_0 = torch.zeros(2 * self.num_layers, 
                              input_seq.shape[0], 
                              self.hidden_size, 
                              device=input_seq.device).detach() # (num_layer, batch_size, hidden_size)
        if c_0 is None:
            c_0 = torch.zeros(2 * self.num_layers, 
                              input_seq.shape[0], 
                              self.hidden_size, 
                              device=input_seq.device).detach() # (num_layer, batch_size, hidden_size)
            
        output, _, _ = self.lstm(input_seq, h_0, c_0) # (batch_size, seq_len, 2 * hidden_size)
        batch_size, seq_len, _ = output.shape
        output = output.view(batch_size * seq_len, 2 * self.hidden_size) # (batch_size * seq_len, 2 * hidden_size)
        output = self.fc(output)
        output = output.view(batch_size, seq_len, self.output_size)
        
        return output
