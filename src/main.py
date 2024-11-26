import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.preprocess import Preprocess
from utils.dataset import TranslationDataset, SentimentDataset
from lstm.lstmcell import LSTMCell, LSTMCell_
from lstm.multi_lstm import MultiLayerLSTM, MultiLayerLSTM_, LSTM, LSTM_, LSTMDecoder
from lstm.bi_lstm import BidirectionalLSTM, BiLSTM
from lstm.seq2seq import Seq2Seq

if __name__ == '__main__':
    # TranslationPreprocess
    print(f"{' TranslationPreprocess ':*^50}")
    dir_path = '../nusax/datasets/mt'
    task = 'machine_translation'
    # emb_type = 'one_hot'
    # preprocess = Preprocess(dir_path, task, emb_type)
    preprocess = Preprocess(dir_path, task)
    preprocess.load_and_preprocess_data()
    print(f"DataFrame Columns: {preprocess.df_train.columns}")
    print(f"Language Vocabulary: {preprocess.vocab.keys()}")
    print('Vocabulary Size:', preprocess.vocab_size['indonesian'])
    print(f"Language Sequence Length: {preprocess.seq_len}")
    print('Vocabulary:')
    cnt = 5
    vocab_indo = preprocess.vocab['indonesian']
    vocab_java = preprocess.vocab['javanese']
    for item1, item2 in zip(vocab_indo.items(), vocab_java.items()):
        if cnt == 0:
            break
        print(f"indonesian: {item1}, \njavanese: {item2}")
        cnt -= 1
    print('Tokenizer:')
    print(eval(preprocess.df_train.loc[0, 'indonesian_tokens']))
    print(eval(preprocess.df_train.loc[0, 'indonesian_word2index']))
    print(eval(preprocess.df_train.loc[0, 'javanese_tokens']))
    print(eval(preprocess.df_train.loc[0, 'javanese_word2index']), '\n')

    # TranslationDataset
    print(f"{' TranslationDataset ':*^50}")
    my_dataset = TranslationDataset(preprocess.df_train, 'indonesian', 'english')
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    print(f"TranslationDataset size: {len(my_dataset)}")
    print(f"Num of Batches: {len(my_dataloader)}")
    print(f"Data: {my_dataset[0]}\n")
    
    for src, tgt in my_dataloader:
        if torch.isnan(src).any() or torch.isinf(src).any():
            print("src contains NaN or inf!")
        if torch.isnan(tgt).any() or torch.isinf(tgt).any():
            print("tgt contains NaN or inf!")
                
    # SentimentPreprocess
    print(f"{' SentimentPreprocess ':*^50}")
    dir_path = '../nusax/datasets/sentiment/indonesian'
    task = 'sentiment_analysis'
    # emb_type = 'one_hot'
    # preprocess = Preprocess(dir_path, task, emb_type)
    preprocess = Preprocess(dir_path, task)
    preprocess.load_and_preprocess_data()
    print(f"DataFrame Columns: preprocess.df_train.columns")
    print(f"Language Vocabulary: preprocess.vocab.keys()")
    print('Vocabulary size:', len(preprocess.vocab['indonesian']))
    print(f"Language Sequence Length: {preprocess.seq_len}")
    print('Vocabulary:')
    cnt = 5
    vocab_indo = preprocess.vocab['indonesian']
    for item1, item2 in zip(vocab_indo.items(), vocab_java.items()):
        if cnt == 0:
            break
        print(f"indonesian: {item1}")
        cnt -= 1
    print('Tokenizer:')
    print(eval(preprocess.df_train.loc[0, 'indonesian_tokens']))
    print(eval(preprocess.df_train.loc[0, 'indonesian_word2index']), '\n')
    
    # SentimentDataset
    print(f"{' SentimentDataset ':*^50}")
    my_dataset = SentimentDataset(preprocess.df_train)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    print(f"SentimentDataset size: {len(my_dataset)}")
    print(f"Num of Batches: {len(my_dataloader)}")
    print(f"Data: {my_dataset[0]}\n")
    
    for text, label in my_dataloader:
        if torch.isnan(text).any() or torch.isinf(text).any():
            print("text contains NaN or inf!")
        if torch.isnan(label).any() or torch.isinf(label).any():
            print("label contains NaN or inf!")
    
    # lstmcell
    print(f"{' LSTMCell ':*^50}")
    batch_size = 4
    x = torch.randn(batch_size, 20, 10)
    lstm=LSTMCell(10, 32)
    h_0=torch.randn(batch_size, 32)
    c_0=torch.randn(batch_size, 32)
    hidden_seq, h, c = lstm(x, h_0, c_0)

    print('hidden_seq.shape:', hidden_seq.shape) # (batch_size, seq_len, hidden_size)
    print('h.shape:', h.shape) # (batch_size, hidden_size)
    print(f'c.shape: {c.shape}\n') # (batch_size, hidden_size)

    # lstmcell
    print(f"{' LSTMCell_ ':*^50}")
    batch_size = 4
    x = torch.randn(batch_size, 10)
    lstm=LSTMCell_(10, 32)
    h_0=torch.randn(batch_size, 32)
    c_0=torch.randn(batch_size, 32)
    h, c = lstm(x, h_0, c_0)

    print('h.shape:', h.shape) # (batch_size, hidden_size)
    print(f'c.shape: {c.shape}\n') # (batch_size, hidden_size)

    # MultiLayerLSTM
    print(f"{' MultiLayerLSTM ':*^50}")
    batch_size = 64
    n_layers = 4
    x = torch.randn(batch_size, 20, 10)
    lstm = MultiLayerLSTM(10, 32, n_layers)
    h_0 = torch.randn(n_layers, batch_size, 32)
    c_0 = torch.randn(n_layers, batch_size, 32)
    output_seq, h, c = lstm(x, h_0, c_0)

    print('hidden_seq.shape:', output_seq.shape) # (hidden_size)
    print('h.shape:', h_0.shape) # (batch_size, hidden_size)
    print(f'c.shape: {c_0.shape}\n') # (batch_size, hidden_size)

    # MultiLayerLSTM_
    print(f"{' MultiLayerLSTM_ ':*^50}")
    batch_size = 64
    n_layers = 4
    x = torch.randn(batch_size, 10)
    lstm = MultiLayerLSTM_(10, 32, n_layers)
    h_0 = torch.randn(n_layers, batch_size, 32)
    c_0 = torch.randn(n_layers, batch_size, 32)
    h, h_t, c_t = lstm(x, h_0, c_0)

    print('hidden_vector.shape:', h.shape) # (batch_size, hidden_size)
    print('h_t.shape:', h_t.shape) # (num_layers, batch_size, hidden_size)
    print(f'c_t.shape: {c_t.shape}\n') # (num_layers, batch_size, hidden_size)
    
    # LSTM
    print(f"{' LSTM ':*^50}")
    output_size = 24
    x = torch.randn(batch_size, 20, 10)
    lstm = LSTM(10, 32, output_size, n_layers)
    output_seq = lstm(x)
    print(f'output.shape: {output_seq.shape}\n', ) # (batch_size, seq_len, output_size)

    # LSTM_
    print(f"{' LSTM_ ':*^50}")
    output_size = 24
    x = torch.randn(batch_size, 10)
    lstm = LSTM_(10, 32, output_size, n_layers)
    output, h_t, c_t = lstm(x)
    print(f'output.shape: {output.shape}') # (batch_size, output_size)
    print('h_t.shape:', h_t.shape) # (num_layers, batch_size, hidden_size)
    print(f'c_t.shape: {c_t.shape}\n') # (num_layers, batch_size, hidden_size)

    # LSTMDecoder
    print(f"{' LSTMDecoder ':*^50}")
    tgt_emb_dim = 32
    hidden_size = 32 
    output_size = 24
    tgt_vocab_dim = 10000
    tgt_embedding = nn.Embedding(tgt_vocab_dim, tgt_emb_dim)
    seq_len = 48
    num_layers = 4
    teacher_forcing_ratio = 0.5
    x = torch.randn(batch_size, seq_len, tgt_emb_dim)
    h_encode = torch.randn(batch_size, hidden_size)
    c_encode = torch.randn(batch_size, hidden_size)
    lstm = LSTMDecoder(tgt_emb_dim, hidden_size, output_size, tgt_embedding, num_layers)
    output = lstm(x, h_encode, c_encode, teacher_forcing_ratio=teacher_forcing_ratio)
    print(f'output.shape: {output.shape}') # (batch_size, seq_len, output_size)
    
    # BidirectionalLSTM
    print(f"{' BidirectionalLSTM ':*^50}")
    batch_size = 64
    n_layers = 4
    x = torch.randn(batch_size, 20, 10)
    lstm = BidirectionalLSTM(10, 32, n_layers)
    h_0 = torch.randn(2 * n_layers, batch_size, 32)
    c_0 = torch.randn(2 * n_layers, batch_size, 32)
    hidden_seq_combined, h_forward, h_backward, c_forward, c_backward = lstm(x, h_0, c_0)

    print('hidden_seq.shape:', hidden_seq_combined.shape) # (batch_size, seq_len, hidden_size)
    print('h_forward.shape:', h_forward.shape) # (batch_size, hidden_size)
    print(f'c_forward.shape: {c_forward.shape}\n') # (batch_size, hidden_size)

    # BiLSTM
    print(f"{' BiLSTM ':*^50}")
    output_size = 24
    lstm = BiLSTM(10, 32, output_size, n_layers)
    output_seq = lstm(x)
    print(f'output.shape: {output_seq.shape}\n') # (batch_size, seq_len, output_size)
    
    # Seq2Seq
    print(f"{' Seq2Seq ':*^50}")
    src_seq_len = 128
    tgt_seq_len = 256
    src_vocab_dim = 2500
    tgt_vocab_dim = 5000
    batch_size = 64
    src_emb_dim = 64 
    tgt_emb_dim = 128
    encoder_hidden_dim = 128 
    decoder_hidden_dim = 128 
    num_encoder_layers = 2 
    num_decoder_layers = 2
    max_len = 500
    teacher_forcing_ratio = 0.5
    mt = Seq2Seq(src_vocab_dim, 
                 tgt_vocab_dim, 
                 src_emb_dim, 
                 tgt_emb_dim, 
                 encoder_hidden_dim, 
                 decoder_hidden_dim,  
                 num_encoder_layers, 
                 num_decoder_layers,
                 max_len,
                 teacher_forcing_ratio)
    src = torch.randint(0, src_vocab_dim, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_dim, (batch_size, tgt_seq_len))
    print('word2index type: ', src[0].dtype)
    output_seq = mt(src, tgt)
    print(f'output.shape: {output_seq.shape}\n') # (batch_size, tgt_seq_len, tgt_vocab_dim)