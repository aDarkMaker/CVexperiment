import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(test_size=0.1, random_state=42):
    """
    加载数据集并进行预处理
    
    Args:
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        X_train, X_test, y_train, y_test: 训练集和测试集的特征和标签
        scaler: 标准化器
    """
    # 读取数据集
    data = pd.read_csv('dateset/dataset.csv', header=None)
    
    # 分离特征和标签
    X = data.iloc[:, :-1].values  # 前两列是特征
    y = data.iloc[:, -1].values   # 最后一列是标签
    
    # 将标签从1,2,3,4转换为0,1,2,3
    y = y - 1
    
    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    """
    创建PyTorch数据加载器
    
    Args:
        X_train, X_test, y_train, y_test: 训练集和测试集
        batch_size: 批量大小
    
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载功能
    X_train, X_test, y_train, y_test, scaler = load_data()
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"类别数量: {len(np.unique(y_train))}")
