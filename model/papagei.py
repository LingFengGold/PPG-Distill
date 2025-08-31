#!/usr/bin/env python
"""
Papagei模型实现
基于ResNet1DMoE架构，适配PPG任务
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logging import info as lprint

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)
        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity
        return out


class ResNet1DMoE(nn.Module):
    """
    ResNet1D with Mixture of Experts (MoE) Regression Heads
    """
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, 
                 n_experts=3, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False,
                 use_projection=False):
        super(ResNet1DMoE, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_projection = use_projection
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.n_experts = n_experts

        # First block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # Residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = (i_block == 0)
            downsample = (i_block % self.downsample_gap == 1)
            
            in_channels = base_filters if is_first_block else int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
            out_channels = in_channels * 2 if (i_block % self.increasefilter_gap == 0 and i_block != 0) else in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=downsample, 
                use_bn=self.use_bn, 
                use_do=self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # Final layers
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)
        
        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            
        # Mixture of Experts (MoE) Head 1
        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, 1)
            ) for _ in range(self.n_experts)
        ])
        self.gating_network_1 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1)
        )

        # Mixture of Experts (MoE) Head 2
        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(out_channels // 2, 1)
            ) for _ in range(self.n_experts)
        ])
        self.gating_network_2 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = x
        
        # First conv layer
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # Residual blocks
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # Final layers
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)  # Global average pooling
        if self.verbose:
            print('final pooling', out.shape)

        if self.use_projection:
            out_class = self.projector(out)
        else:
            out_class = self.dense(out)

        # Mixture of Experts (MoE) Head 1
        expert_outputs_1 = torch.stack([expert(out) for expert in self.expert_layers_1], dim=1)
        gate_weights_1 = self.gating_network_1(out)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)

        # Mixture of Experts (MoE) Head 2
        expert_outputs_2 = torch.stack([expert(out) for expert in self.expert_layers_2], dim=1)
        gate_weights_2 = self.gating_network_2(out)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)

        return out_class, out_moe1, out_moe2, out


def load_model_without_module_prefix(model, checkpoint_path):
    """
    加载模型权重，自动处理module.前缀
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件未找到: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create a new state_dict with the `module.` prefix removed
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_key = k[7:]  # Remove `module.` prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)
    lprint(f"成功加载预训练权重: {checkpoint_path}")
    return model


class PapageiModel(nn.Module):
    """
    适配我们框架的Papagei模型
    加载预训练的ResNet1DMoE并添加任务特定的输出层
    """
    def __init__(self, model_config):
        super(PapageiModel, self).__init__()
        
        # 从配置中获取参数
        self.base_filters = model_config.get('base_filters', 32)
        self.kernel_size = model_config.get('kernel_size', 3)
        self.stride = model_config.get('stride', 2)
        self.groups = model_config.get('groups', 1)
        self.n_block = model_config.get('n_block', 18)
        self.n_classes = model_config.get('n_classes', 512)
        self.n_experts = model_config.get('n_experts', 3)
        self.output_size = model_config.get('output_size', 1)
        self.model_path = model_config.get('model_path', 'papagei-foundation-model/weights/papagei_s.pt')
        self.freeze_backbone = model_config.get('freeze_backbone', False)
        self.dropout = model_config.get('dropout', 0.2)
        
        # 创建ResNet1DMoE骨干网络
        self.backbone = ResNet1DMoE(
            in_channels=1, 
            base_filters=self.base_filters, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            n_block=self.n_block,
            n_classes=self.n_classes,
            n_experts=self.n_experts
        )
        
        # 加载预训练权重
        if os.path.exists(self.model_path):
            self.backbone = load_model_without_module_prefix(self.backbone, self.model_path)
        else:
            lprint(f"警告: 预训练模型文件未找到: {self.model_path}")
            lprint("将使用随机初始化的权重")
        
        # 冻结骨干网络（如果指定）
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            lprint("已冻结骨干网络参数")
        
        # 添加任务特定的输出层
        # 计算特征维度（ResNet1DMoE的最后一层特征维度）
        feature_dim = self._get_feature_dim()
        
        # 创建分类器/回归器
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(feature_dim // 2, self.output_size)
        )
        
        lprint(f"Papagei模型初始化完成:")
        lprint(f"  - 骨干网络: ResNet1DMoE ({self.n_block} blocks)")
        lprint(f"  - 特征维度: {feature_dim}")
        lprint(f"  - 输出维度: {self.output_size}")
        lprint(f"  - 冻结骨干网络: {self.freeze_backbone}")
    
    def _get_feature_dim(self):
        """计算特征维度"""
        # ResNet1DMoE的最后一层特征维度计算
        # 根据n_block和increasefilter_gap计算最终的out_channels
        increasefilter_gap = 4  # 默认值
        out_channels = self.base_filters
        
        for i_block in range(self.n_block):
            if i_block == 0:
                in_channels = self.base_filters
                out_channels = in_channels
            else:
                # increase filters at every increasefilter_gap blocks
                in_channels = int(self.base_filters * 2 ** ((i_block - 1) // increasefilter_gap))
                if (i_block % increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
        
        return out_channels
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, n_patches, patch_size) 或 (batch_size, 1, sequence_length)
        
        Returns:
            output: 任务输出 (batch_size, output_size)
        """
        # 确保输入格式正确
        if x.dim() == 3:
            if x.size(1) != 1:
                # 如果输入是 (batch_size, n_patches, patch_size)，需要重塑为 (batch_size, 1, sequence_length)
                batch_size, n_patches, patch_size = x.shape
                x = x.view(batch_size, 1, n_patches * patch_size)
            # 如果输入已经是 (batch_size, 1, sequence_length) 格式，保持不变
        elif x.dim() == 2:
            # 如果输入是 (batch_size, sequence_length)，添加通道维度
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")
            
        # 调试信息
        # lprint(f"Papagei模型输入形状: {x.shape}")
        
        # 通过骨干网络提取特征
        with torch.set_grad_enabled(not self.freeze_backbone):
            out_class, out_moe1, out_moe2, features = self.backbone(x)
        
        # 使用特征进行最终预测
        output = self.classifier(features)
        
        return output
    
    def extract_patch_features(self, x):
        """
        提取patch级别特征，用于知识蒸馏
        
        Args:
            x: 输入张量 (batch_size, n_patches, patch_size)
        
        Returns:
            patch_features: patch级别特征 (batch_size, n_patches, feature_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"extract_patch_features需要3维输入 (batch_size, n_patches, patch_size)，但得到: {x.shape}")
        
        batch_size, n_patches, patch_size = x.shape
        
        # 将每个patch单独处理
        patch_features_list = []
        
        for i in range(n_patches):
            # 提取第i个patch: (batch_size, patch_size)
            patch_i = x[:, i, :]  # (batch_size, patch_size)
            
            # 添加通道维度: (batch_size, 1, patch_size)
            patch_i = patch_i.unsqueeze(1)
            
            # 通过骨干网络提取特征
            with torch.set_grad_enabled(not self.freeze_backbone):
                out_class, out_moe1, out_moe2, features = self.backbone(patch_i)
            
            # features: (batch_size, feature_dim)
            patch_features_list.append(features)
        
        # 堆叠所有patch的特征: (batch_size, n_patches, feature_dim)
        patch_features = torch.stack(patch_features_list, dim=1)
        
        return patch_features
    
    def get_global_features(self, x):
        """
        获取全局特征，用于知识蒸馏
        
        Args:
            x: 输入张量 (batch_size, n_patches, patch_size) 或 (batch_size, 1, sequence_length)
        
        Returns:
            features: 全局特征 (batch_size, feature_dim)
        """
        # 确保输入格式正确
        if x.dim() == 3:
            if x.size(1) != 1:
                # 如果输入是 (batch_size, n_patches, patch_size)，需要重塑为 (batch_size, 1, sequence_length)
                batch_size, n_patches, patch_size = x.shape
                x = x.view(batch_size, 1, n_patches * patch_size)
            # 如果输入已经是 (batch_size, 1, sequence_length) 格式，保持不变
        elif x.dim() == 2:
            # 如果输入是 (batch_size, sequence_length)，添加通道维度
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")
        
        # 通过骨干网络提取特征
        with torch.set_grad_enabled(not self.freeze_backbone):
            out_class, out_moe1, out_moe2, features = self.backbone(x)
        
        return features


def create_papagei_model(model_config):
    """
    创建Papagei模型的工厂函数
    
    Args:
        model_config: 模型配置字典
    
    Returns:
        PapageiModel: 配置好的Papagei模型
    """
    return PapageiModel(model_config)


# 模型参数计算函数
def count_papagei_parameters(model):
    """计算Papagei模型的参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    lprint(f"Papagei模型参数统计:")
    lprint(f"  - 总参数数: {total_params:,}")
    lprint(f"  - 可训练参数数: {trainable_params:,}")
    lprint(f"  - 骨干网络参数数: {backbone_params:,}")
    lprint(f"  - 分类器参数数: {classifier_params:,}")
    
    return total_params, trainable_params 