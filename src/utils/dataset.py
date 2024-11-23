import torch
import pandas as pd
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, source_lang: str, target_lang: str):
        self.source_sentences = data[f"{source_lang}_word2index"].to_list()
        self.target_sentences = data[f"{target_lang}_word2index"].to_list()
    
    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, idx):
        source_sentence = eval(self.source_sentences[idx])
        target_sentence = eval(self.target_sentences[idx])
        return torch.tensor(source_sentence), torch.tensor(target_sentence)

class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.sentences = data[f"indonesian_word2index"].to_list()
        self.labels = data[f"label"].to_list()
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = eval(self.sentences[idx])
        label = self.labels[idx]
        return torch.tensor(sentence), torch.tensor(label)
