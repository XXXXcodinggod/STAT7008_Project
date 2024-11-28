import torch
import torch.nn.functional as F
from tqdm import tqdm

def TranslationTrain(model, 
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
                     max_norm,
                     checkpoint_path):
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    initial_teacher_forcing_ratio = 1.0
    
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0 
        teacher_forcing_ratio = max(min_teacher_forcing_ratio, 
                                    initial_teacher_forcing_ratio - decay_rate * epoch)
        print(f"teacher_forcing_ratio: {teacher_forcing_ratio}")
        for src, tgt in tqdm(train_loader, desc="Training", unit="batch"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output, _ = model(src, tgt[:, :-1], teacher_forcing_ratio, temperature)  # use preceding token to predict succeeding token
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1)) # compare predicted tokens and ground truth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Successfully save the model to {checkpoint_path}")
    
    return train_losses, valid_losses

def SentimentTrain(model, train_loader, valid_loader, device, criterion, optimizer, epochs, checkpoint_path):
    train_losses = []
    train_accs = []
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Successfully save the model to {checkpoint_path}")
    
    return train_losses, train_accs
