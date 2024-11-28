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
                 num_layers,
                 dropout_rate,
                 max_len):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_dim, src_emb_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_dim, tgt_emb_dim)
        self.encoder = BidirectionalLSTM(src_emb_dim, encoder_hidden_dim, num_layers, dropout_rate)
        self.mid_layer_h = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
        self.mid_layer_c = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
        self.decoder = LSTMDecoder(tgt_emb_dim, 
                                   decoder_hidden_dim, 
                                   tgt_vocab_dim, 
                                   self.tgt_embedding, 
                                   num_layers,
                                   dropout_rate)
        self.max_len = max_len # maximum decoding length
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0, temperature=1.0):
        batch_size = src.size(0) # (batch_size, seq_len)
        src_emb = self.src_embedding(src) # (batch_size, seq_len, src_emb_dim)
        _, h_combined, c_combined = self.encoder(src_emb) # (num_layers, batch_size, 2 * encoder_hidden_dim)
        h_encode = self.mid_layer_h(h_combined) # （num_layers, batch_size, decoder_hidden_dim)
        c_encode = self.mid_layer_c(c_combined) # （num_layers, batch_size, decoder_hidden_dim)
        
        # print('-'*50)
        # h_encode, c_encode = h_encode.transpose(0, 1), c_encode.transpose(0, 1)
        # print('状态编码')
        # print(h_encode[:5,:,:4])
        # # print('细胞编码')
        # # print(c_encode[:5,:,:4])
        # # print('-'*50)
        # h_encode, c_encode = h_encode.transpose(0, 1), c_encode.transpose(0, 1)
        
        if self.training:
            tgt_emb = self.tgt_embedding(tgt) # (batch_size, seq_len - 1, tgt_emb_dim)
        else:
            # intialize with ['START']: 2; (batch_size, max_len, tgt_emb_dim)
            teacher_forcing_ratio = 0
            tgt = torch.full((batch_size,), 2, device=src.device).detach()
            tgt_emb = self.tgt_embedding(tgt).unsqueeze(1).expand(-1, self.max_len, -1)
        outputs, predicted_tokens = self.decoder(tgt_emb, h_encode, c_encode, teacher_forcing_ratio, temperature)

        return outputs, predicted_tokens
