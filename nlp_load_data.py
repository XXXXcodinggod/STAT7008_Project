import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import re

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

if __name__=='__main__':
    nltk.download('punkt')
    nltk.download('punkt_tab')

    #######################################机器翻译##########################################################
    ## 机器学习
    path = 'nusax/datasets/mt/'
    train_df = pd.read_csv(path + 'train.csv', index_col=0)[['indonesian', 'english', 'javanese']]
    valid_df = pd.read_csv(path + 'valid.csv', index_col=0)[['indonesian', 'english', 'javanese']]

    # Tokenize the datasets
    in_train_arr, in_dict = tokenize(train_df['indonesian'])
    en_train_arr, en_dict = tokenize(train_df['english'])
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

    #######################################情感分析##########################################################
    ## 机器学习
    path = 'nusax/datasets/sentiment/indonesian/'
    xtrain, ytrain = load_data(path + 'train.csv')
    xvalid, yvalid = load_data(path + 'valid.csv')
    xtest, ytest = load_data(path + 'test.csv')

    ## 深度学习
    train_df = pd.read_csv(path + 'train.csv')
    valid_df = pd.read_csv(path + 'valid.csv')
    # Tokenize the datasets
    train_text, train_dict = tokenize(train_df['text'])
    valid_text, valid_dict = tokenize(valid_df['text'])
    # encoding labels
    y_train = sentiment_encode(train_df['label'])
    y_valid = sentiment_encode(valid_df['label'])

    # Pad sequences
    max_len = 128
    train_text_pad = padding(train_text, max_len)
    valid_text_pad = padding(valid_text, max_len)

    # Create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_text_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(valid_text_pad), torch.from_numpy(y_valid))

    # Create DataLoaders
    batch_size = 32
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)



