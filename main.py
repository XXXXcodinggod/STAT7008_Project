import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src import Preprocess, TranslationDataset, SentimentDataset
from src import SentimentTrain, TranslationTrain, plot_losses
from src import Seq2Seq

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config('run.yaml')
    print(f"config: {config}\n")
    
    data_path = config['data_path']
    task = config['task']
    src_lang = config['src_lang'] 
    tgt_lang = config['tgt_lang']
    model_config = config['model']
    train_config = config['train']
    output_config = config['output']
    model_name = model_config['model']
    src_emb_dim = model_config['src_emb_dim']
    tgt_emb_dim = model_config['tgt_emb_dim']
    encoder_hidden_dim = model_config['encoder_hidden_dim']
    decoder_hidden_dim = model_config['decoder_hidden_dim']
    num_encoder_layers = model_config['num_encoder_layers']
    num_decoder_layers = model_config['num_decoder_layers']
    max_len = model_config['max_len']
    teacher_forcing_ratio = model_config['teacher_forcing_ratio']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']
    momentum = train_config['momentum']
    checkpoint_path = output_config['checkpoint_path']
    plot_path = output_config['plot_path']
    
    preprocess = Preprocess(data_path, task)
    preprocess.load_and_preprocess_data()
    src_vocab_dim = preprocess.vocab_size[src_lang]
    tgt_vocab_dim = preprocess.vocab_size[tgt_lang]
    train_dataset = TranslationDataset(preprocess.df_train, src_lang, tgt_lang)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TranslationDataset(preprocess.df_valid, src_lang, tgt_lang)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    if model_name == 'Seq2Seq':
        print((src_vocab_dim, 
               tgt_vocab_dim, 
               src_emb_dim, 
               tgt_emb_dim, 
               encoder_hidden_dim, 
               decoder_hidden_dim, 
               tgt_vocab_dim, 
               num_encoder_layers, 
               num_decoder_layers,
               teacher_forcing_ratio))
        model = Seq2Seq(src_vocab_dim, 
                        tgt_vocab_dim, 
                        src_emb_dim, 
                        tgt_emb_dim, 
                        encoder_hidden_dim, 
                        decoder_hidden_dim, 
                        num_encoder_layers, 
                        num_decoder_layers,
                        max_len,
                        teacher_forcing_ratio).to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    train_losses = TranslationTrain(model, 
                                    train_loader, 
                                    valid_loader, 
                                    device, 
                                    criterion, 
                                    optimizer, 
                                    scheduler, 
                                    epochs, 
                                    checkpoint_path)
    plot_losses(train_losses, plot_path)
    
if __name__ == '__main__':
    main()
