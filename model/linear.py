"""
Linear模型定义
简单的线性模型用于PPG信号处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    """简单的线性模型"""
    
    def __init__(self, config):
        super(LinearModel, self).__init__()
        self.config = config
        
        # 计算输入特征维度
        patch_size = config['patch_size']
        n_patches = config['n_patches']
        input_dim = patch_size * n_patches
        
        # 输出维度
        output_dim = config['output_size']
        
        # 创建线性层
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Dropout层（可选）
        self.dropout = nn.Dropout(config.get('dropout', 0.0))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """前向传播"""
        # x shape: [batch_size, n_patches, patch_size]
        batch_size = x.size(0)
        
        # 展平输入
        x = x.view(batch_size, -1)  # [batch_size, n_patches * patch_size]
        
        # 应用dropout
        x = self.dropout(x)
        
        # 线性变换
        output = self.linear(x)
        
        return output
    
    def get_features(self, x):
        """获取特征表示（线性模型直接返回输入特征）"""
        batch_size = x.size(0)
        return x.view(batch_size, -1)
    
    def get_num_parameters(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())


class LinearWithActivation(nn.Module):
    """带激活函数的线性模型"""
    
    def __init__(self, config):
        super(LinearWithActivation, self).__init__()
        self.config = config
        
        # 计算输入特征维度
        patch_size = config['patch_size']
        n_patches = config['n_patches']
        input_dim = patch_size * n_patches
        
        # 隐藏层维度
        hidden_dim = config.get('hidden_dim', 512)
        
        # 输出维度
        output_dim = config['output_size']
        
        # 创建层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        activation = config.get('activation', 'relu').lower()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Dropout层
        self.dropout = nn.Dropout(config.get('dropout', 0.0))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.input_layer.bias is not None:
            nn.init.zeros_(self.input_layer.bias)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """前向传播"""
        # x shape: [batch_size, n_patches, patch_size]
        batch_size = x.size(0)
        
        # 展平输入
        x = x.view(batch_size, -1)  # [batch_size, n_patches * patch_size]
        
        # 隐藏层
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output
    
    def get_features(self, x):
        """获取隐藏层特征"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 获取隐藏层特征
        features = self.input_layer(x)
        features = self.activation(features)
        
        return features
    
    def get_num_parameters(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())


def create_linear_model(config):
    """创建线性模型工厂函数"""
    use_activation = config.get('use_activation', False)
    
    if use_activation:
        model = LinearWithActivation(config)
    else:
        model = LinearModel(config)
    
    print(f"线性模型配置: {config}")
    print(f"模型参数数量: {model.get_num_parameters():,}")
    
    return model 