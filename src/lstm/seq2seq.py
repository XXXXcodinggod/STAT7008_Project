import torch
import torch.nn as nn
from .multi_lstm import LSTM
from .bi_lstm import BidirectionalLSTM

class Seq2Seq(nn.Module):
    def __init__(self, src_seq_len, tgt_seq_len, src_emb_dim, tgt_emb_dim, encoder_hidden_dim, decoder_hidden_dim, tgt_vocab_dim, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.src_embedding = nn.Embedding(src_seq_len, src_emb_dim)
        self.tgt_embedding = nn.Embedding(tgt_seq_len, tgt_emb_dim)
        self.encoder = BidirectionalLSTM(src_emb_dim, encoder_hidden_dim, num_encoder_layers)
        self.mid_layer_h = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
        self.mid_layer_c = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
        self.decoder = LSTM(tgt_emb_dim, decoder_hidden_dim, tgt_vocab_dim, num_decoder_layers)
    
    def forward(self, src, tgt):
        src_emb = self.src_embedding(src)
        _, h_forward, h_backward, c_forward, c_backward = self.encoder(src_emb)
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        c_combined = torch.cat([c_forward, c_backward], dim=1)
        h_encode = self.mid_layer_h(h_combined) 
        c_encode = self.mid_layer_c(c_combined)
        tgt_emb = self.tgt_embedding(tgt)
        output = self.decoder(tgt_emb, h_encode, c_encode)
        return output