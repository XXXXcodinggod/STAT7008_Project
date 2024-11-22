import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.lstm.multi_lstm import LSTM
from src.lstm.bi_lstm import BiLSTM

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 训练函数
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
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

# 准备保存训练过程的图像
def save_plots(losses, accs, epochs):
    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accs, label='Training Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

# 读取 YAML 配置文件
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 主程序
if __name__ == '__main__':
    # 加载配置
    config = load_config('train.yaml')
    model_config = config['model']
    print(model_config)
    
    # 创建模型
    if model_config['task'] == 'mt':
        if model_config['model'] == 'lstm':
            model = LSTM(model_config['input_size'], 
                         model_config['hidden_size'], 
                         model_config['output_size'])
    batch_size = 64
    n_layers = 4
    output_size = 24
    x = torch.randn(batch_size, 20, 10)
    h_0 = torch.randn(n_layers, batch_size, 32)
    c_0 = torch.randn(n_layers, batch_size, 32)
    output_seq = model(x)
    print('output.shape:', output_seq.shape) # (batch_size, seq_len, output_size)
    
    # # 创建假数据
    # x_train = torch.randn(1000, config['model']['input_size'])
    # y_train = torch.randint(0, config['model']['output_size'], (1000,))

    # # 创建数据集和数据加载器
    # dataset = TensorDataset(x_train, y_train)
    # dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # # 训练模型
    # train_losses, train_accs = train_model(model, dataloader, criterion, optimizer, config['training']['epochs'])

    # # 保存训练过程图像
    # save_plots(train_losses, train_accs, config['training']['epochs'])