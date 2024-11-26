import seaborn as sns
import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losses, plot_path):
    """
    绘制训练损失和验证损失的折线图
    
    参数:
    train_losses (list): 训练损失列表
    valid_losses (list): 验证损失列表
    plot_path     (str): 图片保存路径
    """
    sns.set_style(style='darkgrid')
    epochs = list(range(1, len(train_losses) + 1))  # 生成 [1, 2, ..., n]
    plt.figure(figsize=(10, 6))  # 设置图形大小
    sns.lineplot(x=epochs, y=train_losses, label='Training Loss', marker='o')  # 训练损失
    sns.lineplot(x=epochs, y=valid_losses, label='Validation Loss', marker='o')  # 验证损失
    plt.title('Training and Validation Loss per Epoch')  # 图形标题
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')  # y轴标签
    plt.legend()  # 显示图例
    plt.grid()  # 添加网格
    plt.xticks(epochs)  # 设置x轴刻度
    plt.tight_layout()  # 调整布局
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot successfully saved to {plot_path}")
    plt.show()  # 显示图形