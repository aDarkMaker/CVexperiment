import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from util.data_load import load_data, create_dataloaders

class FeedForwardNN(nn.Module):
    """
    前馈神经网络模型
    """
    def __init__(self, input_size=2, hidden_layers=[64], output_size=4, activation='relu'):
        """
        初始化神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_layers: 隐藏层神经元数量列表，例如[64]表示单隐藏层64个神经元，[128, 64]表示双隐藏层
            output_size: 输出类别数量
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        super(FeedForwardNN, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())  # 默认使用ReLU
                
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    """
    训练模型并记录损失和准确率
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
    
    Returns:
        train_losses: 每个mini-batch的训练损失
        test_losses: 每个epoch的测试损失
        train_accuracies: 每个epoch的训练准确率
        test_accuracies: 每个epoch的测试准确率
    """
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_losses.append(loss.item())  # 记录每个batch的损失
            
            # 计算训练准确率
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # 计算训练准确率
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss/len(train_loader):.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, test_losses, train_accuracies, test_accuracies

def evaluate_model(model, train_loader, test_loader, criterion):
    """
    评估模型在训练集和测试集上的最终性能
    """
    model.eval()
    
    # 训练集评估
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            train_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    train_loss /= len(train_loader)
    
    # 测试集评估
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    test_loss /= len(test_loader)
    
    return train_loss, train_accuracy, test_loss, test_accuracy

def plot_results(train_losses, test_losses, train_accuracies, test_accuracies, config_name):
    """
    绘制训练结果图表
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Train Loss (per batch)', alpha=0.6)
    ax1.plot(np.linspace(0, len(train_losses)-1, len(test_losses)), test_losses, 
             label='Test Loss (per epoch)', linewidth=2)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{config_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{config_name} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results_{config_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_experiment(config, X_train, X_test, y_train, y_test):
    """
    运行单个实验配置
    """
    print(f"\n=== 实验配置: {config['name']} ===")
    print(f"网络架构: {config['hidden_layers']}")
    print(f"激活函数: {config['activation']}")
    print(f"学习率: {config['learning_rate']}")
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=config.get('batch_size', 32)
    )
    
    # 创建模型
    model = FeedForwardNN(
        input_size=2,
        hidden_layers=config['hidden_layers'],
        output_size=4,
        activation=config['activation']
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练模型
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        num_epochs=config.get('epochs', 100)
    )
    
    # 最终评估
    train_loss, train_acc, test_loss, test_acc = evaluate_model(
        model, train_loader, test_loader, criterion
    )
    
    print(f"最终结果 - 训练集准确率: {train_acc:.2f}%, 测试集准确率: {test_acc:.2f}%")
    
    # 绘制结果
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, config['name'])
    
    return {
        'config': config,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def main():
    """
    主函数：运行多组实验
    """
    # 加载数据
    X_train, X_test, y_train, y_test, scaler = load_data()
    print(f"数据加载完成: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
    
    # 定义实验配置
    experiment_configs = [
        {
            'name': '单隐藏层64节点+ReLU',
            'hidden_layers': [64],
            'activation': 'relu',
            'learning_rate': 0.001,
            'epochs': 100
        },
        {
            'name': '双隐藏层128-64节点+ReLU',
            'hidden_layers': [128, 64],
            'activation': 'relu',
            'learning_rate': 0.001,
            'epochs': 100
        },
        {
            'name': '单隐藏层32节点+Tanh',
            'hidden_layers': [32],
            'activation': 'tanh',
            'learning_rate': 0.001,
            'epochs': 100
        },
        {
            'name': '单隐藏层64节点+Sigmoid',
            'hidden_layers': [64],
            'activation': 'sigmoid',
            'learning_rate': 0.001,
            'epochs': 100
        },
        {
            'name': '高学习率单隐藏层64节点+ReLU',
            'hidden_layers': [64],
            'activation': 'relu',
            'learning_rate': 0.01,
            'epochs': 100
        }
    ]
    
    # 运行所有实验
    results = []
    for config in experiment_configs:
        result = run_experiment(config, X_train, X_test, y_train, y_test)
        results.append(result)
    
    # 打印实验总结
    print("\n" + "="*50)
    print("实验总结报告")
    print("="*50)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['config']['name']}:")
        print(f"   训练集准确率: {result['train_accuracy']:.2f}%")
        print(f"   测试集准确率: {result['test_accuracy']:.2f}%")
        print(f"   网络架构: {result['config']['hidden_layers']}")
        print(f"   激活函数: {result['config']['activation']}")
        print(f"   学习率: {result['config']['learning_rate']}")
        print()
    
    # 找出最佳配置
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print(f"最佳配置: {best_result['config']['name']}")
    print(f"最佳测试准确率: {best_result['test_accuracy']:.2f}%")

if __name__ == "__main__":
    main()
