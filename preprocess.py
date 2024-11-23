import pandas as pd
import os
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize

current_directory = os.getcwd()
nltk.data.path.append(current_directory)
nltk.download('punkt', download_dir=current_directory)
nltk.download('punkt_tab', download_dir=current_directory)

class Preprocess:
    def __init__(self, dir_path: str, task: str, emb_type: str):
        self.dir_path = dir_path
        self.task = task
        self.emb_type = emb_type
        self.df_train = None
        self.df_test = None
        self.df_valid = None
        self.seq_len = {}
        self.vocab = {}
        if self.task == 'machine_translation':
            self.languages = ['indonesian', 'javanese', 'english']  
        elif self.task == 'sentiment_analysis':
            self.languages = ['indonesian']
        
    def _vocabulary(self):
        for language in self.languages:
            word_list = []
            for tokens in self.df_train[f"{language}_tokens"]:
                tokens = eval(tokens)
                word_list += tokens
            corpus = Counter(word_list)
            # corpus = sorted(corpus, key=corpus.get, reverse=True)[:1000]
            corpus = sorted(corpus, key=corpus.get, reverse=True)
            vocabulary = {w: i+1 for i, w in enumerate(corpus)}
            vocabulary['[UNK]'] = 0
            self.vocab[language] = vocabulary
        
    def _tokenize(self):
        df_train = self.df_train
        df_test = self.df_test
        df_valid = self.df_valid
        
        if self.task == 'sentiment_analysis':
            self.df_train.rename(columns={'text': 'indonesian'}, inplace=True)
            self.df_test.rename(columns={'text': 'indonesian'}, inplace=True)
            self.df_valid.rename(columns={'text': 'indonesian'}, inplace=True)
        
        for language in self.languages:
            df_train[f"{language}_tokens"] = df_train[language].apply(lambda x: str(word_tokenize(x.lower())))
            df_test[f"{language}_tokens"] = df_test[language].apply(lambda x: str(word_tokenize(x.lower())))
            df_valid[f"{language}_tokens"] = df_valid[language].apply(lambda x: str(word_tokenize(x.lower())))
            self.seq_len[language] = max([len(eval(tokens)) for tokens in df_train[f"{language}_tokens"].tolist()])
            
        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid
    
    def _sentiment_encode(self):
        label_dict = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        df_train = self.df_train
        df_test = self.df_test
        df_valid = self.df_valid
        
        df_train["label"] = df_train["label"].apply(lambda x: label_dict[x])
        df_test["label"] = df_test["label"].apply(lambda x: label_dict[x])
        df_valid["label"] = df_valid["label"].apply(lambda x: label_dict[x])
    
        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid
    
    def _unify_seq_len(self):
        df_train = self.df_train
        df_test = self.df_test
        df_valid = self.df_valid
        
        def _padding_truncating(seq_len, tokens):
            tokens = eval(tokens)
            token_len = len(tokens)
            if token_len == seq_len:
                return str(tokens)
            elif token_len < seq_len:
                return str(tokens + ['[PAD]'] * (seq_len - len(tokens)))
            else:
                return str(tokens[:seq_len])
        
        for language in self.languages:
            seq_len = self.seq_len[language]
            df_train[f"{language}_tokens"] = df_train[f"{language}_tokens"].apply(lambda x: _padding_truncating(seq_len, x))
            df_test[f"{language}_tokens"] = df_test[f"{language}_tokens"].apply(lambda x: _padding_truncating(seq_len, x))
            df_valid[f"{language}_tokens"] = df_valid[f"{language}_tokens"].apply(lambda x: _padding_truncating(seq_len, x))
        
        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid
        
    def _one_hot(self):
        df_train = self.df_train
        df_test = self.df_test
        df_valid = self.df_valid
        
        def _one_hot_embedding(vocab, seq_len, tokens):
            one_hot_embedding = [[0] * (len(vocab))] * seq_len
            word_to_index = [vocab[token] if token in vocab else 1 for token in eval(tokens)]
            for i in range(seq_len):
                one_hot_embedding[i][word_to_index[i]] = 1
            return str(one_hot_embedding)
        
        for language in self.languages:
            vocab = self.vocab[language]
            seq_len = self.seq_len[language]
            
            df_train[f"{language}_embedding"] = df_train[f"{language}_tokens"].apply(lambda x: _one_hot_embedding(vocab, seq_len, x))
            df_test[f"{language}_embedding"] = df_test[f"{language}_tokens"].apply(lambda x: _one_hot_embedding(vocab, seq_len, x))
            df_valid[f"{language}_embedding"] = df_valid[f"{language}_tokens"].apply(lambda x: _one_hot_embedding(vocab, seq_len, x))

        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid

    def _embedding(self):
        pass
    
    def load_and_preprocess_data(self):
        self.df_train = pd.read_csv(os.path.join(self.dir_path, 'train.csv'))
        self.df_test = pd.read_csv(os.path.join(self.dir_path, 'test.csv'))
        self.df_valid = pd.read_csv(os.path.join(self.dir_path, 'valid.csv'))
        
        self._tokenize()
        self._unify_seq_len()
        if self.task == 'sentiment_analysis':
            self._sentiment_encode()
        if self.emb_type == 'one_hot':
            self._vocabulary()
            self._one_hot()
        elif self.emb_type == 'word2vec':
            pass
        elif self.emb_type == 'bert':
            pass            
        
if __name__ ==  '__main__':
    # dir_path = './nusax/datasets/sentiment/indonesian'
    # task = 'sentiment_analysis'
    dir_path = './nusax/datasets/mt'
    task = 'machine_translation'
    emb_type = 'one_hot'
    preprocess = Preprocess(dir_path, task, emb_type)
    preprocess.load_and_preprocess_data()
    print(preprocess.vocab.keys())
    print('vocabulary size:', len(preprocess.vocab['indonesian']))
    print(preprocess.seq_len)
    for item in preprocess.vocab['indonesian'].items():
        print(item)
        break
    print('sequence length:', len(eval(preprocess.df_train.loc[0, 'indonesian_embedding'])))
    print('one_hot_embed_dim:', len(eval(preprocess.df_train.loc[0, 'indonesian_embedding'])[0]))