import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import re
import torch.utils.data as Data

from nltk import word_tokenize
import nltk

#确保已导入文件夹，否则命令行执行 git clone https://github.com/IndoNLP/nusax.git

def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)  # Remove punctuation
    s = re.sub(r"\s+", ' ', s).strip()  # Remove extra spaces
    return s

def tokenize(data):
    word_list = []
    for sent in data:
        words = str(sent).lower().split()
        for word in words:
            word = preprocess_string(word)
            if word:
                word_list.append(word)

    # Create frequency distribution
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)    # 情感分析时，可以排除频次较小的次，比如保留前1000个
    word_to_index = {w: i + 1 for i, w in enumerate(corpus_)}  # Index starts from 1

    # Tokenize sentences
    tokenized_data = []
    for sent in data:
        tokens = []
        words = str(sent).lower().split()
        for word in words:
            word = preprocess_string(word)
            if word and word in word_to_index:
                tokens.append(word_to_index[word])
        tokenized_data.append(tokens if tokens else [0])  # Add padding token if empty

    return np.array(tokenized_data, dtype=object), word_to_index

def padding(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for i, sent in enumerate(sentences):
        if len(sent) > 0:
            sent_arr = np.array(sent)
            features[i, -len(sent_arr):] = sent_arr[-max_len:]  # Pad from the end
    return features

def load_data(filedir):
    df = pd.read_csv(filedir)
    data = list(df['text'])
    data = [" ".join(word_tokenize(sent)) for sent in data]
    return (data, list(df['label']))


def sentiment_encode(labels):
    # Encoding strategy: pos = 2, neu = 1, neg = 0
    encoded_labels = [2 if label == 'positive' else (1 if label == 'neutral' else 0) for label in labels]
    return np.array(encoded_labels)


nltk.download('punkt')
nltk.download('punkt_tab')

path = 'datasets/mt/'
train_df = pd.read_csv(path + 'train.csv', index_col=0)[['indonesian', 'english', 'javanese']]
#print(train_df.head(5))
valid_df = pd.read_csv(path + 'valid.csv', index_col=0)[['indonesian', 'english', 'javanese']]
#print(valid_df.head(5))

# Tokenize the datasets
in_train_arr, in_dict = tokenize(train_df['indonesian'])
#print(in_train_arr)
#print(in_dict)
en_train_arr, en_dict = tokenize(train_df['english'])
#print(en_train_arr)
#print(en_dict)
ja_train_arr, ja_dict = tokenize(train_df['javanese'])

in_valid_arr, _ = tokenize(valid_df['indonesian'])
en_valid_arr, _ = tokenize(valid_df['english'])
ja_valid_arr, _ = tokenize(valid_df['javanese'])

## 深度学习
# Pad sequences
max_len = 128
in_train_pad = padding(in_train_arr, max_len)
en_train_pad = padding(en_train_arr, max_len)
ja_train_pad = padding(ja_train_arr, max_len)

in_valid_pad = padding(in_valid_arr, max_len)
en_valid_pad = padding(en_valid_arr, max_len)
ja_valid_pad = padding(ja_valid_arr, max_len)

# Create Tensor datasets
train_data_en = TensorDataset(torch.from_numpy(in_train_pad), torch.from_numpy(en_train_pad))
train_data_ja = TensorDataset(torch.from_numpy(in_train_pad), torch.from_numpy(ja_train_pad))

# Create DataLoaders
batch_size = 32
train_loader_en = DataLoader(train_data_en, shuffle=True, batch_size=batch_size)
train_loader_ja = DataLoader(train_data_ja, shuffle=True, batch_size=batch_size)

# P represents placeholder, S represents starter, E represents terminator
sentences = [['saya pelajar P', 'S am a orange', 'am a orange E'],
             ['saya suka belajar', 'S tiny chocolate P', 'tiny chocolate P E'],
             ['saya saat P', 'S whether able believe', 'whether able believe E']]

in_dict['P'] = 0
en_dict['the'] = 2858
en_dict['and'] = 2859
en_dict['I'] = 2860
en_dict['P'] = 0
en_dict['S'] = 1
en_dict['E'] = 2
in_idx2word = {in_dict[key]: key for key in in_dict}
src_vocab = in_dict
src_vocab_size = len(in_dict)
en_idx2word = {en_dict[key]: key for key in en_dict}
tgt_vocab = en_dict
tgt_vocab_size = len(en_dict)
src_len = len(sentences[0][0].split(" "))
tgt_len = len(sentences[0][1].split(" "))

# Transform to dictionaries
def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# Define datasets
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]