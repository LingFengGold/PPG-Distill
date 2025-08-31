import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """
    多层感知机模型，用于PPG信号回归任务
    """
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: list = [512, 256, 128],
                 output_size: int = 1,
                 dropout: float = 0.2,
                 activation: str = 'relu',
                 batch_norm: bool = True,
                 **kwargs):
        """
        初始化MLP模型
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出维度（回归任务通常为1）
            dropout: dropout率
            activation: 激活函数类型 ('relu', 'gelu', 'tanh')
            batch_norm: 是否使用批归一化
        """
        super().__init__()
        
        if kwargs:
            print(f"Ignored keyword arguments: {kwargs}")
            
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # 线性层
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 批归一化
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # 激活函数
            layers.append(self.activation)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, patch_size] 或 [batch_size, input_size]
        
        Returns:
            输出张量，形状为 [batch_size, output_size]
        """
        # 如果输入是3D张量（批次，序列，特征），展平为2D
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)  # 展平所有特征
        
        return self.network(x)
    
    def get_features(self, x):
        """
        获取模型的特征表示（倒数第二层的输出）
        
        Args:
            x: 输入张量
            
        Returns:
            特征张量
        """
        # 如果输入是3D张量，展平为2D
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        
        # 通过除了最后一层的所有层
        for layer in self.network[:-1]:
            x = layer(x)
        
        return x
    
    def get_num_parameters(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)





def create_mlp_model(config):
    """
    根据配置创建MLP模型的工厂函数
    
    Args:
        config: 模型配置字典
    
    Returns:
        MLP模型实例
    """
    # 计算输入大小
    patch_size = config.get('patch_size', 40)
    n_patches = config.get('n_patches', 60)
    input_size = patch_size * n_patches
    
    # 创建MLP配置，移除不相关的参数
    mlp_config = {
        'input_size': input_size,
        'hidden_sizes': config.get('hidden_sizes', [512, 256, 128]),
        'output_size': config.get('output_size', 1),
        'dropout': config.get('dropout', 0.2),
        'activation': config.get('activation', 'relu'),
        'batch_norm': config.get('batch_norm', True)
    }

    print(f'mlp_config: {mlp_config}')
    
    return MLP(**mlp_config)


 