import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src import Preprocess, TranslationDataset, SentimentDataset
from src import SentimentTrain, TranslationTrain, TranslationTest, plot_losses
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
    num_layers = model_config['num_layers']
    dropout_rate = model_config['dropout_rate']
    max_len = model_config['max_len']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']
    momentum = train_config['momentum']
    weight_decay = train_config['weight_decay']
    min_teacher_forcing_ratio = train_config['min_teacher_forcing_ratio']
    temperature = train_config['temperature']
    decay_rate = train_config['decay_rate']
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
    test_dataset = TranslationDataset(preprocess.df_test, src_lang, tgt_lang)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    if model_name == 'Seq2Seq':
        model = Seq2Seq(src_vocab_dim, 
                        tgt_vocab_dim, 
                        src_emb_dim, 
                        tgt_emb_dim, 
                        encoder_hidden_dim, 
                        decoder_hidden_dim, 
                        num_layers, 
                        dropout_rate,
                        max_len).to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1) 
    train_losses, valid_losses = TranslationTrain(model, 
                                                  train_loader, 
                                                  valid_loader, 
                                                  device, 
                                                  criterion, 
                                                  optimizer, 
                                                  scheduler, 
                                                  epochs,
                                                  min_teacher_forcing_ratio,
                                                  temperature,
                                                  decay_rate,
                                                  1.0,
                                                  checkpoint_path)
    
    model = Seq2Seq(src_vocab_dim, 
                    tgt_vocab_dim, 
                    src_emb_dim, 
                    tgt_emb_dim, 
                    encoder_hidden_dim, 
                    decoder_hidden_dim, 
                    num_layers,
                    dropout_rate,
                    max_len).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss = TranslationTest(model, test_loader, device, criterion, preprocess, tgt_lang, temperature)
    print(test_loss)

    print('train src')
    print(preprocess.df_train[f"{src_lang}_tokens"].head(20))
    print(preprocess.df_train[f"{src_lang}_word2index"].head(20))
    print(preprocess.df_train[f"{src_lang}_tokens"].loc[0][-100:])
    print(preprocess.df_train[f"{src_lang}_word2index"].loc[0][-100:])
    
    print('train tgt')
    print(preprocess.df_train[f"{tgt_lang}_tokens"].head(20))
    print(preprocess.df_train[f"{tgt_lang}_word2index"].head(20))
    print(preprocess.df_train[f"{tgt_lang}_tokens"].loc[0][-100:])
    print(preprocess.df_train[f"{tgt_lang}_word2index"].loc[0][-100:])
    
    print('test src')
    print(preprocess.df_test[f"{src_lang}_tokens"].head(20))
    print(preprocess.df_test[f"{src_lang}_word2index"].head(20))
    
    print('test tgt')
    print(preprocess.df_test[f"{tgt_lang}_tokens"].head(20))
    print(preprocess.df_test[f"{tgt_lang}_word2index"].head(20))
    print(preprocess.df_train[f"{tgt_lang}_tokens"].loc[0][-100:])
    print(preprocess.df_train[f"{tgt_lang}_word2index"].loc[0][-100:])
    
    cnt = 5
    for index, row in preprocess.df_test.iterrows():
        predicted_tokens = row[f"{tgt_lang}_predicted_tokens"]
        predicted_sentence = row[f"{tgt_lang}_predicted_sentence"]
        if cnt == 0:
            break
        if '[END]' in predicted_tokens:
            print('-'*50)
            print(preprocess.df_test.loc[index, f"{tgt_lang}"])
            print(predicted_tokens)
            print(predicted_sentence)
            print('-'*50)
            cnt -= 1
            
if __name__ == '__main__':
    main()
