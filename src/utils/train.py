import torch

def SentimentTrain(model, dataloader, device, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    train_accs = []
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
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

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, train_accs

def TranslationTrain(model, dataloader, device, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        epoch_loss = 0 
        
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            if torch.isnan(src).any() or torch.isinf(src).any():
                print("src contains NaN or inf!")
            if torch.isnan(tgt).any() or torch.isinf(tgt).any():
                print("tgt contains NaN or inf!")
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # use preceding token to predict succeeding token
            print(output)
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1)) # compare predicted tokens and ground truth
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

    return train_losses

