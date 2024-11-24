import pandas as pd
import os
import nltk
from typing import Optional
from collections import Counter
from nltk.tokenize import word_tokenize

class Preprocess:
    def __init__(self, dir_path: str, task: str, emb_type: Optional[str] = None):
        self.dir_path = dir_path
        self.task = task
        self.emb_type = emb_type
        self.df_train = None
        self.df_test = None
        self.df_valid = None
        self.seq_len = {}
        self.vocab = {}
        self.vocab_size = {}
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
            vocabulary = {'[PAD]': 0, '[UNK]':1}
            if language in ['javanese', 'english']:
                vocabulary['[START]'] = 2
                vocabulary['[END]'] = 3
            n = len(vocabulary)
            for i, word in enumerate(corpus):
                vocabulary[word] = n + i
            self.vocab[language] = vocabulary
            self.vocab_size[language] = len(vocabulary)
        
    def _tokenize(self):
        nltk.data.path.append('./utils') # from src directory
        nltk.data.path.append('./src/utils') # from root directory
        
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
        
        def _padding_truncating(seq_len, tokens, is_target_lang):
            tokens = eval(tokens)
            if is_target_lang:
                tokens = ['[START]'] + tokens + ['[END]']
            token_len = len(tokens)
            if token_len == seq_len:
                return str(tokens)
            elif token_len < seq_len:
                return str(tokens + ['[PAD]'] * (seq_len - len(tokens)))
            else:
                if not is_target_lang:
                    return str(tokens[:seq_len])
                else:
                    return str(tokens[:seq_len - 1] + ['[END]'])
        
        for language in self.languages:
            seq_len = self.seq_len[language]
            is_target_lang = 0 if language == 'indonesian' else 1
            df_train[f"{language}_tokens"] = df_train[f"{language}_tokens"].apply(lambda x: _padding_truncating(seq_len, x, is_target_lang))
            df_test[f"{language}_tokens"] = df_test[f"{language}_tokens"].apply(lambda x: _padding_truncating(seq_len, x, is_target_lang))
            df_valid[f"{language}_tokens"] = df_valid[f"{language}_tokens"].apply(lambda x: _padding_truncating(seq_len, x, is_target_lang))
        
        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid
    
    def _word_to_index(self):
        df_train = self.df_train
        df_test = self.df_test
        df_valid = self.df_valid
        
        def _mapping(vocab, tokens):
            return str([vocab[token] if token in vocab else 1 for token in eval(tokens)])
        
        for language in self.languages:
            vocab = self.vocab[language]
            df_train[f"{language}_word2index"] = df_train[f"{language}_tokens"].apply(lambda x: _mapping(vocab, x))
            df_test[f"{language}_word2index"] = df_test[f"{language}_tokens"].apply(lambda x: _mapping(vocab, x))
            df_valid[f"{language}_word2index"] = df_valid[f"{language}_tokens"].apply(lambda x: _mapping(vocab, x))

        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid

    # def _one_hot(self):
    #     df_train = self.df_train
    #     df_test = self.df_test
    #     df_valid = self.df_valid
        
    #     def _one_hot_embedding(vocab, word2index):
    #         word2index = eval(word2index)
    #         one_hot_embedding = [[0] * (len(vocab))] * len(word2index)
    #         for i in range(seq_len):
    #             one_hot_embedding[i][word2index[i]] = 1
    #         return str(one_hot_embedding)
        
    #     for language in self.languages:
    #         vocab = self.vocab[language]
    #         seq_len = self.seq_len[language]
            
    #         df_train[f"{language}_embedding"] = df_train[f"{language}_word2index"].apply(lambda x: _one_hot_embedding(vocab, x))
    #         df_test[f"{language}_embedding"] = df_test[f"{language}_word2index"].apply(lambda x: _one_hot_embedding(vocab, x))
    #         df_valid[f"{language}_embedding"] = df_valid[f"{language}_word2index"].apply(lambda x: _one_hot_embedding(vocab, x))

    #     self.df_train = df_train
    #     self.df_test = df_test
    #     self.df_valid = df_valid

    def _embedding(self):
        pass
    
    def load_and_preprocess_data(self):
        self.df_train = pd.read_csv(os.path.join(self.dir_path, 'train.csv'))
        self.df_test = pd.read_csv(os.path.join(self.dir_path, 'test.csv'))
        self.df_valid = pd.read_csv(os.path.join(self.dir_path, 'valid.csv'))
        
        self._tokenize()
        self._vocabulary()
        self._unify_seq_len()
        self._word_to_index()
        if self.task == 'sentiment_analysis':
            self._sentiment_encode()
        
        # if self.emb_type == 'one_hot':
        #     self._one_hot()
        # elif self.emb_type == 'word2vec':
        #     pass
        # elif self.emb_type == 'bert':
        #     pass            

if __name__ == '__main__':
    current_directory = os.getcwd()
    nltk.download('punkt', download_dir=current_directory)
    nltk.download('punkt_tab', download_dir=current_directory)
    