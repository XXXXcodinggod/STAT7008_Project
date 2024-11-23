import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from preprocess import Preprocess

class TranslationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, source_lang: str, target_lang: str):
        self.source_sentences = data[f"{source_lang}_embedding"].to_list()
        self.target_sentences = data[f"{target_lang}_embedding"].to_list()
    
    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, idx):
        source_sentence = eval(self.source_sentences[idx])
        target_sentence = eval(self.target_sentences[idx])
        return torch.tensor(source_sentence), torch.tensor(target_sentence)

class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.sentences = data[f"indonesian_embedding"].to_list()
        self.labels = data[f"label"].to_list()
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = eval(self.sentences[idx])
        label = self.labels[idx]
        return torch.tensor(sentence), torch.tensor(label)

if __name__ ==  '__main__':
    dir_path = './nusax/datasets/mt'
    task = 'machine_translation'
    emb_type = 'one_hot'
    preprocess = Preprocess(dir_path, task, emb_type)
    preprocess.load_and_preprocess_data()
    my_dataset = TranslationDataset(preprocess.df_train, 'indonesian', 'english')
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    print(f"TranslationDataset size: {len(my_dataset)}")
    print(f"Num of Batches: {len(my_dataloader)}")

    dir_path = './nusax/datasets/sentiment/indonesian'
    task = 'sentiment_analysis'
    emb_type = 'one_hot'
    preprocess = Preprocess(dir_path, task, emb_type)
    preprocess.load_and_preprocess_data()
    my_dataset = SentimentDataset(preprocess.df_train)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    print(f"SentimentDataset size: {len(my_dataset)}")
    print(f"Num of Batches: {len(my_dataloader)}")