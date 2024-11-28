import torch
import torch.nn.functional as F

def TranslationTest(model, test_loader, device, criterion, preprocess, tgt_lang, temperature):
    vocab = {index: token for token, index in preprocess.vocab[tgt_lang].items()}
    df_test = preprocess.df_test
    output_tokens = []
    output_sentences = []
    
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for src, tgt in test_loader: # src: (batch_size, src_seq_len)
            src, tgt = src.to(device), tgt.to(device)
            output, predicted_token = model(src, temperature=temperature) # outputs: (batch_size, max_len, tgt_vocab_dim)
            tgt_seq_len, max_len = tgt.size(1), output.size(1)
            if tgt_seq_len < max_len: # tgt_seq_len < max_len
                tgt = F.pad(tgt, (0, max_len - tgt_seq_len), value=0) # ['PAD']: 0
            elif tgt_seq_len > max_len: # tgt_seq_len > max_len
                tgt = tgt[:, :max_len] # truncate extra part due to model decoding capacity limitations
            loss = criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
            test_loss += loss.item()
            
            predicted_token = predicted_token.tolist() # type list, (batch_size, max_len)
            for sequence in predicted_token:
                predicted_tokens = [vocab[index] for index in sequence]
                output_tokens.append(str(predicted_tokens))
                output_sentences.append(' '.join(predicted_tokens))

    df_test[f"{tgt_lang}_predicted_tokens"] = output_tokens
    df_test[f"{tgt_lang}_predicted_sentence"] = output_sentences
    preprocess.df_test = df_test
    avg_test_loss = test_loss / len(test_loader)
    
    return avg_test_loss