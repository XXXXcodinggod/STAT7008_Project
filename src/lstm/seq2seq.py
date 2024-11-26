import torch
import torch.nn as nn
from .multi_lstm import LSTMDecoder, LSTM
from .bi_lstm import BidirectionalLSTM

class Seq2Seq(nn.Module):
    def __init__(self, 
                 src_vocab_dim, 
                 tgt_vocab_dim, 
                 src_emb_dim, 
                 tgt_emb_dim, 
                 encoder_hidden_dim, 
                 decoder_hidden_dim, 
                 num_encoder_layers, 
                 num_decoder_layers,
                 max_len,
                 teacher_forcing_ratio):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_dim, src_emb_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_dim, tgt_emb_dim)
        self.encoder = BidirectionalLSTM(src_emb_dim, encoder_hidden_dim, num_encoder_layers)
        self.mid_layer_h = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
        self.mid_layer_c = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
        self.decoder = LSTMDecoder(tgt_emb_dim, 
                                   decoder_hidden_dim, 
                                   tgt_vocab_dim, 
                                   self.tgt_embedding, 
                                   num_decoder_layers)
        self.max_len = max_len # maximum decoding length
        self.teacher_forcing_ratio = teacher_forcing_ratio # probabilty of teacher forcing
        
    def forward(self, src, tgt=None):
        batch_size = src.size(0)
        src_emb = self.src_embedding(src)
        _, h_forward, h_backward, c_forward, c_backward = self.encoder(src_emb)
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        c_combined = torch.cat([c_forward, c_backward], dim=1)
        h_encode = self.mid_layer_h(h_combined) 
        c_encode = self.mid_layer_c(c_combined)
        if self.training:
            teacher_forcing_ratio = self.teacher_forcing_ratio
            tgt_emb = self.tgt_embedding(tgt) 
        else:
            teacher_forcing_ratio = 0
            # intialize with ['START']: 2; (batch_size, max_len, tgt_emb_dim)
            tgt = torch.full((batch_size,), 2, device=src.device)
            tgt_emb = self.tgt_embedding(tgt).unsqueeze(1).expand(-1, self.max_len, -1)
        output = self.decoder(tgt_emb, h_encode, c_encode, teacher_forcing_ratio)
        return output
