#!/usr/bin/env python
"""
统一的PPG模型知识蒸馏训练脚本
支持GPT、Linear、MLP作为teacher或student模型的各种组合
"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import sys
sys.path.append('.')
import argparse
import torch
import logging
logging.basicConfig(level=logging.INFO)
from logging import info as lprint
import yaml
import json
import numpy as np
import tqdm
import random
import time
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import csv

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.nn as nn
import torch.nn.functional as F

# 导入距离计算函数
def pdist(e, squared=False, eps=1e-12):
    """计算成对距离"""
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    
    if not squared:
        res = res.sqrt()
    
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    """关系知识蒸馏 - 距离损失"""
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


class LrScheduler:
    """自定义学习率调度器，支持warmup + cosine annealing"""
    def __init__(self, optimizer, warmup_scheduler, main_scheduler, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0

    def get_last_lr(self):
        if self.step_count < self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.main_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.step_count < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
        self.step_count += 1

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.main_scheduler.load_state_dict(state_dict['main_scheduler'])

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'main_scheduler': self.main_scheduler.state_dict(),
        }

# 导入模型
from model.gpt import GPT_with_linearOutput
from model.linear import LinearModel, create_linear_model
from model.mlp import MLP, create_mlp_model
from model.papagei import PapageiModel, create_papagei_model, count_papagei_parameters
from data.pretrain_dataset import PretrainDataset
try:
    from local.supp_fxns import *
except ImportError:
    pass  # 如果导入失败，使用内建功能


def calc_n_patches(filepath, fs):
    """计算数据的patch数量"""
    assert os.path.isfile(filepath), f'{filepath=}'
    seq_length = get_numpy_array_metadata(filepath, return_attrs=True)[0][-1]
    assert seq_length >= fs, 'need atleast 1 second signal length'
    assert seq_length % fs == 0, 'signal length (in seconds) must be a whole number'
    n_patches = seq_length // fs
    return n_patches


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, temperature=4.0, 
                 use_feature_distill=True, feature_loss_weight=1.0,
                 use_patch_feature_distill=False, patch_feature_loss_weight=1.0,
                 patch_distill_mode='direct', contrastive_temperature=0.1,
                 use_patch_distance_distill=False, patch_distance_loss_weight=1.0,
                 # 降参优化选项
                 tie_feature_and_patch=True):    # 合并表征蒸馏
        """
        Args:
            alpha: 预测蒸馏损失权重（可扫参数）
            beta: 全局特征蒸馏损失权重（可扫参数）
            gamma: patch级别蒸馏损失权重（可扫参数，包含patch_feature_loss和patch_distance_loss）
            temperature: 软化温度
            use_feature_distill: 是否使用全局特征蒸馏
            feature_loss_weight: 全局特征蒸馏损失权重
            use_patch_feature_distill: 是否使用patch级别特征蒸馏
            patch_feature_loss_weight: patch特征蒸馏损失权重
            patch_distill_mode: patch蒸馏模式 ('direct' 或 'contrastive')
            contrastive_temperature: 对比学习温度参数 (或 'auto')
            use_patch_distance_distill: 是否使用patch距离蒸馏
            patch_distance_loss_weight: patch距离蒸馏损失权重
            tie_feature_and_patch: 是否合并表征蒸馏
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.use_feature_distill = use_feature_distill
        self.feature_loss_weight = feature_loss_weight
        self.use_patch_feature_distill = use_patch_feature_distill
        self.patch_feature_loss_weight = patch_feature_loss_weight
        self.patch_distill_mode = patch_distill_mode
        self.contrastive_temperature = contrastive_temperature
        self.use_patch_distance_distill = use_patch_distance_distill
        self.patch_distance_loss_weight = patch_distance_loss_weight
        
        # 降参优化选项
        self.tie_feature_and_patch = tie_feature_and_patch
        
        # 特征适配器（全局和patch特征共用，因为都来自同一个backbone）
        self.feature_adapter = None
        
        # patch距离蒸馏器
        self.patch_distance_distiller = RkdDistance() if use_patch_distance_distill else None
        
    def setup_feature_adapter(self, teacher_dim, student_dim, teacher_patch_dim=None, student_patch_dim=None):
        """设置特征适配器，全局和patch特征共用同一个适配器（因为维度相同）"""
        
        # 检查维度一致性
        if teacher_patch_dim is not None and teacher_patch_dim != teacher_dim:
            lprint(f"警告: teacher全局特征维度({teacher_dim})与patch特征维度({teacher_patch_dim})不匹配")
        
        if student_patch_dim is not None and student_patch_dim != student_dim:
            lprint(f"警告: student全局特征维度({student_dim})与patch特征维度({student_patch_dim})不匹配")
        
        # 只要teacher和student的特征维度不匹配就需要适配器
        need_adapter = (
            (self.use_feature_distill and teacher_dim != student_dim) or
            (self.use_patch_feature_distill and teacher_dim != student_dim)
        )
        
        if need_adapter:
            self.feature_adapter = nn.Linear(teacher_dim, student_dim)
            lprint(f"创建共享特征适配器: {teacher_dim} -> {student_dim} (全局和patch特征共用)")
        else:
            lprint("特征维度匹配，无需适配器")
    

        
    def forward(self, student_output, teacher_output, student_features=None, 
                teacher_features=None, targets=None, task_type='regression',
                student_patch_features=None, teacher_patch_features=None):
        """
        计算蒸馏损失（支持降参优化）
        
        Args:
            student_output: 学生模型输出
            teacher_output: 教师模型输出  
            student_features: 学生模型全局特征（可选）
            teacher_features: 教师模型全局特征（可选）
            targets: 真实标签（可选）
            task_type: 任务类型
            student_patch_features: 学生模型patch级别特征 [B, n_patches, dim]（可选）
            teacher_patch_features: 教师模型patch级别特征 [B, n_patches, dim]（可选）
        """
        
        # === 步骤1: 计算原始损失分量 ===
        gt_loss = None
        if targets is not None:
            if task_type == 'regression':
                gt_loss = F.mse_loss(student_output.squeeze(), targets)
            else:
                gt_loss = F.cross_entropy(student_output, targets.long())

        if task_type == 'regression':
            pred_loss = F.mse_loss(student_output, teacher_output)
        else:
            teacher_soft = F.softmax(teacher_output / self.temperature, dim=1)
            student_log_soft = F.log_softmax(student_output / self.temperature, dim=1)
            pred_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
            pred_loss = pred_loss * (self.temperature ** 2)

        # 全局特征蒸馏损失
        global_feat_loss = None
        if self.use_feature_distill and student_features is not None and teacher_features is not None:
            if len(teacher_features.shape) == 3:
                teacher_features = teacher_features.mean(dim=1)
            teacher_feat_adapted = self.feature_adapter(teacher_features) if self.feature_adapter is not None else teacher_features
            global_feat_loss = F.mse_loss(student_features, teacher_feat_adapted)

        # patch级别特征蒸馏损失
        patch_feat_loss = None
        if self.use_patch_feature_distill and student_patch_features is not None and teacher_patch_features is not None:
            # 应用特征适配器（需要正确处理3D张量）
            if self.feature_adapter is not None:
                teacher_patch_adapted = self.feature_adapter(teacher_patch_features)
            else:
                teacher_patch_adapted = teacher_patch_features
            
            if self.patch_distill_mode == 'direct':
                patch_feat_loss = F.mse_loss(student_patch_features, teacher_patch_adapted)
            else:
                patch_feat_loss = self._compute_contrastive_patch_loss(student_patch_features, teacher_patch_adapted)

        # patch距离蒸馏损失
        patch_distance_loss = None
        if self.use_patch_distance_distill and student_patch_features is not None and teacher_patch_features is not None:
            # 对每个batch中的patch进行距离蒸馏
            batch_size = student_patch_features.size(0)
            patch_distance_losses = []
            
            for b in range(batch_size):
                student_patches_b = student_patch_features[b]  # [n_patches, dim]
                teacher_patches_b = teacher_patch_features[b]  # [n_patches, dim]
                
                # 计算patch之间的距离蒸馏损失
                patch_dist_loss_b = self.patch_distance_distiller(student_patches_b, teacher_patches_b)
                patch_distance_losses.append(patch_dist_loss_b)
            
            patch_distance_loss = torch.stack(patch_distance_losses).mean()

        # === 步骤2: 直接使用原始损失（无EMA归一化） ===
        ngt = gt_loss
        npred = pred_loss
        nfeat = global_feat_loss
        npatch = patch_feat_loss
        npatch_distance = patch_distance_loss

        # === 步骤3: 合并patch级别蒸馏（如果启用） ===
        npatch_combined = None
        if self.tie_feature_and_patch:
            # 合并patch级别的损失
            patch_parts = [x for x in [npatch, npatch_distance] if x is not None]
            npatch_combined = sum(patch_parts) if patch_parts else None

        # === 步骤4: 组合损失 ===
        loss_dict = {}
        
        # 新的权重模式：gt_loss=1, pred_loss=alpha, global_feat_loss=beta, patch_loss=gamma
        total = 0.0
        if ngt is not None:
            total += 1.0 * ngt  # gt_loss权重固定为1
        total += self.alpha * npred  # pred_loss权重为alpha
        
        if self.tie_feature_and_patch:
            # 合并模式：全局特征用beta，patch级别用gamma
            if nfeat is not None:
                total += self.beta * nfeat  # 全局特征蒸馏权重为beta
            if npatch_combined is not None:
                total += self.gamma * npatch_combined  # patch级别蒸馏权重为gamma
        else:
            # 分离模式：各自独立权重
            if nfeat is not None:
                total += self.beta * nfeat  # 全局特征蒸馏权重为beta
            if npatch is not None:
                total += self.gamma * npatch  # patch特征蒸馏权重为gamma
            if npatch_distance is not None:
                total += self.gamma * npatch_distance  # patch距离蒸馏权重为gamma

        loss_dict['gt_loss'] = ngt if ngt is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['pred_distill_loss'] = npred
        loss_dict['feature_distill_loss'] = nfeat if nfeat is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['patch_feature_distill_loss'] = npatch if npatch is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['patch_distance_distill_loss'] = npatch_distance if npatch_distance is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['total_loss'] = total
        return loss_dict
    
    def _compute_patch_distillation_loss(self, student_patch_features, teacher_patch_features):
        """
        计算patch级别特征蒸馏损失
        
        Args:
            student_patch_features: [B, n_patches, dim_s]
            teacher_patch_features: [B, n_patches, dim_t]
            
        Returns:
            patch_distill_loss: 标量损失值
        """
        # 应用特征适配器（需要正确处理3D张量）
        if self.feature_adapter is not None:
            # teacher_patch_features: [B, n_patches, teacher_dim] -> [B, n_patches, student_dim]
            B, n_patches, teacher_dim = teacher_patch_features.shape
            # 重塑为2D张量以应用线性变换
            teacher_patch_flat = teacher_patch_features.view(B * n_patches, teacher_dim)
            teacher_patch_adapted_flat = self.feature_adapter(teacher_patch_flat)
            # 重塑回3D张量
            student_dim = teacher_patch_adapted_flat.shape[-1]
            teacher_patch_features_adapted = teacher_patch_adapted_flat.view(B, n_patches, student_dim)
        else:
            teacher_patch_features_adapted = teacher_patch_features
        
        if self.patch_distill_mode == 'direct':
            # 直接匹配模式：对应patch之间计算MSE损失
            patch_distill_loss = F.mse_loss(student_patch_features, teacher_patch_features_adapted)
            
        elif self.patch_distill_mode == 'contrastive':
            # 对比学习模式：对应patch为正样本，其他patch为负样本
            patch_distill_loss = self._compute_contrastive_patch_loss(
                student_patch_features, teacher_patch_features_adapted
            )
        else:
            raise ValueError(f"不支持的patch蒸馏模式: {self.patch_distill_mode}")
        
        return patch_distill_loss
    
    def _compute_contrastive_patch_loss(self, student_patches, teacher_patches):
        """
        计算对比学习的patch蒸馏损失
        
        Args:
            student_patches: [B, n_patches, dim]
            teacher_patches: [B, n_patches, dim]
            
        Returns:
            contrastive_loss: 标量损失值
        """
        batch_size, n_patches, dim = student_patches.shape
        
        # 自动温度设置
        if isinstance(self.contrastive_temperature, str) and self.contrastive_temperature == 'auto':
            tau = 1.0 / math.sqrt(dim)
        else:
            tau = self.contrastive_temperature
        
        # 将student和teacher patch特征归一化
        student_patches_norm = F.normalize(student_patches, dim=-1)  # [B, n_patches, dim]
        teacher_patches_norm = F.normalize(teacher_patches, dim=-1)  # [B, n_patches, dim]
        
        total_loss = 0.0
        
        # 对每个batch中的每个patch进行对比学习
        for b in range(batch_size):
            student_b = student_patches_norm[b]  # [n_patches, dim]
            teacher_b = teacher_patches_norm[b]   # [n_patches, dim]
            
            # 计算student patch与所有teacher patch的相似度
            # similarities: [n_patches, n_patches]
            similarities = torch.mm(student_b, teacher_b.t()) / tau
            
            # 对角线元素为正样本（对应patch），其余为负样本
            labels = torch.arange(n_patches, device=similarities.device)
            
            # 使用交叉熵损失
            contrastive_loss_b = F.cross_entropy(similarities, labels)
            total_loss += contrastive_loss_b
        
        # 平均所有batch的损失
        return total_loss / batch_size


class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, teacher_model, student_model, train_loader, val_loader, 
                 config, device, save_path, task_type='regression', test_loader=None):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.save_path = save_path
        self.task_type = task_type
        
        # 创建蒸馏损失
        distill_config = config['distillation']
        self.distill_loss = DistillationLoss(
            alpha=distill_config.get('alpha', 0.5),
            beta=distill_config.get('beta', 0.5),
            gamma=distill_config.get('gamma', 0.5),
            temperature=distill_config.get('temperature', 4.0),
            use_feature_distill=distill_config.get('use_feature_distill', True),
            feature_loss_weight=distill_config.get('feature_loss_weight', 1.0),
            use_patch_feature_distill=distill_config.get('use_patch_feature_distill', False),
            patch_feature_loss_weight=distill_config.get('patch_feature_loss_weight', 1.0),
            patch_distill_mode=distill_config.get('patch_distill_mode', 'direct'),
            contrastive_temperature=distill_config.get('contrastive_temperature', 0.1),
            use_patch_distance_distill=distill_config.get('use_patch_distance_distill', False),
            patch_distance_loss_weight=distill_config.get('patch_distance_loss_weight', 1.0),
            # 降参优化选项
            tie_feature_and_patch=distill_config.get('tie_feature_and_patch', True)
        )
        
        # 设置特征适配器
        teacher_dim = self.get_feature_dim(teacher_model)
        student_dim = self.get_feature_dim(student_model)
        
        # 获取patch级别特征维度（仅对GPT模型）
        teacher_patch_dim = None
        student_patch_dim = None
        if self.distill_loss.use_patch_feature_distill:
            teacher_patch_dim = self.get_patch_feature_dim(teacher_model)
            student_patch_dim = self.get_patch_feature_dim(student_model)
        
        self.distill_loss.setup_feature_adapter(teacher_dim, student_dim, teacher_patch_dim, student_patch_dim)
        
        # 将特征适配器移动到正确的设备
        if self.distill_loss.feature_adapter is not None:
            self.distill_loss.feature_adapter = self.distill_loss.feature_adapter.to(device)
            lprint(f"特征适配器已移动到设备: {device}")
        
        # 将patch距离蒸馏器移动到正确的设备
        if self.distill_loss.patch_distance_distiller is not None:
            self.distill_loss.patch_distance_distiller = self.distill_loss.patch_distance_distiller.to(device)
            lprint(f"patch距离蒸馏器已移动到设备: {device}")
        
        # 创建优化器参数列表
        optimizer_params = list(self.student_model.parameters())
        
        # 添加共享的特征适配器参数
        if self.distill_loss.feature_adapter is not None:
            optimizer_params.extend(self.distill_loss.feature_adapter.parameters())
            lprint("特征适配器参数已加入优化器")
        
        # 优化器
        train_config = config['train_config']
        optimizer_type = train_config.get('optimizer', 'Adam').lower()
        # 使用初始学习率作为优化器的学习率，后续通过调度器调整
        lr = float(train_config.get('lr_init', 1e-5))
        weight_decay = float(train_config.get('weight_decay', 0.0))
        
        if optimizer_type == 'adam':
            self.optimizer = Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 学习率调度器 - 按batch粒度调度
        self.scheduler = None
        
        # 获取学习率相关配置，设置默认值并确保类型正确
        lr_init = float(train_config.get('lr_init', 1e-5))
        lr_max = float(train_config.get('lr_max', 30e-5))
        lr_final = float(train_config.get('lr_final', 1e-6))
        lr_schedule_ratio = float(train_config.get('lr_schedule_ratio', 1))
        lr_warm_up = float(train_config.get('lr_warm_up', 0.25))
        
        # 计算总的调度步数（按batch计算）
        epochs = train_config['epochs']
        batches_per_epoch = len(train_loader)
        lr_schedule_step = int(lr_schedule_ratio * epochs * batches_per_epoch)
        warm_up_steps = int(lr_schedule_step * lr_warm_up)
        
        lprint(f"学习率调度配置:")
        lprint(f"  lr_init: {lr_init}, lr_max: {lr_max}, lr_final: {lr_final}")
        lprint(f"  总调度步数: {lr_schedule_step}, 预热步数: {warm_up_steps}")
        lprint(f"  每epoch批次数: {batches_per_epoch}")
        
        # 创建预热调度器
        lambda_warmup = lambda step: (lr_init + step * (lr_max - lr_init) / warm_up_steps) / lr_init
        warm_up_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_warmup)
        
        # 创建主调度器（余弦退火）
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=lr_schedule_step - warm_up_steps, eta_min=lr_final)
        
        # 创建组合调度器
        self.scheduler = LrScheduler(self.optimizer, warm_up_scheduler, main_scheduler, warm_up_steps, lr_schedule_step)
        
        lprint(f"已创建按batch粒度的学习率调度器 (warmup + cosine annealing)")
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # 早停参数
        self.early_stop_patience = train_config.get('early_stop_patience', 20)  # 默认20个epoch无改善就停止
        self.early_stop_min_delta = float(train_config.get('early_stop_min_delta', 1e-6))  # 最小改善阈值
        self.early_stop_counter = 0
        self.early_stopped = False
        
        # 性能统计
        self.all_train_batch_times = []
        self.all_val_batch_times = []
        self.all_test_batch_times = []
        
    def get_feature_dim(self, model):
        """获取模型特征维度 - 通过实际前向传播探测"""
        # 使用一个小batch的数据来探测特征维度
        model.eval()
        with torch.no_grad():
            # 获取一个小批次的数据进行探测
            probe = next(iter(self.train_loader))["ppg_segments"].to(self.device)[:2]
            features = self.get_features(model, probe)
            
            if features.dim() == 3:
                feature_dim = features.shape[-1]
            else:
                feature_dim = features.shape[-1]
            
            lprint(f"探测到模型全局特征维度: {feature_dim}")
            return feature_dim
    
    def get_patch_feature_dim(self, model):
        """获取模型patch级别特征维度 - 适用于GPT模型和Papagei模型"""
        if hasattr(model, 'gpt'):
            # GPT模型
            model.eval()
            with torch.no_grad():
                # 获取一个小批次的数据进行探测
                probe = next(iter(self.train_loader))["ppg_segments"].to(self.device)[:2]
                patch_features = self.get_patch_features(model, probe)
                
                if patch_features is None:
                    return None
                
                patch_feature_dim = patch_features.shape[-1]  # [B, n_patches, dim]
                lprint(f"探测到GPT模型patch特征维度: {patch_feature_dim}")
                return patch_feature_dim
        elif isinstance(model, PapageiModel):
            # Papagei模型
            model.eval()
            with torch.no_grad():
                # 获取一个小批次的数据进行探测
                probe = next(iter(self.train_loader))["ppg_segments"].to(self.device)[:2]
                patch_features = self.get_patch_features(model, probe)
                
                if patch_features is None:
                    return None
                
                patch_feature_dim = patch_features.shape[-1]  # [B, n_patches, dim]
                lprint(f"探测到Papagei模型patch特征维度: {patch_feature_dim}")
                return patch_feature_dim
        else:
            lprint(f"模型 {type(model).__name__} 不支持patch级别特征蒸馏")
            return None
    
    def get_features(self, model, x):
        """获取模型的特征表示"""
        if isinstance(model, PapageiModel):
            # Papagei模型：使用专门的全局特征提取方法
            if model == self.teacher_model:
                with torch.no_grad():
                    features = model.get_global_features(x)
            else:
                features = model.get_global_features(x)
            return features
        elif hasattr(model, 'gpt'):
            # GPT模型：获取编码特征
            if model == self.teacher_model:
                with torch.no_grad():
                    features = model.gpt.encode(x, apply_mask=False)
            else:
                features = model.gpt.encode(x, apply_mask=False)
            
            if features.dim() == 3:
                features = features.mean(dim=1)  # 平均池化 [B, S, D] -> [B, D]
            return features
        elif hasattr(model, 'get_features'):
            # 如果模型有专门的特征提取方法
            return model.get_features(x)
        else:
            # Linear/MLP模型：使用中间层输出作为特征
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            
            if hasattr(model, 'layers') and len(model.layers) > 1:
                # MLP模型：获取最后一个隐藏层
                features = x_flat
                for i, layer in enumerate(model.layers[:-1]):
                    features = layer(features)
                    if hasattr(model, 'activation') and i < len(model.layers) - 2:
                        features = model.activation(features)
                return features
            else:
                # Linear模型：直接返回输入特征
                return x_flat
    
    def get_patch_features(self, model, x):
        """获取模型的patch级别特征 - 适用于GPT模型和Papagei模型"""
        if hasattr(model, 'gpt'):
            # GPT模型
            if model == self.teacher_model:
                with torch.no_grad():
                    # 通过GPT编码器获取patch级别特征
                    patch_features = model.gpt.encode(x, apply_mask=False)  # [B, n_patches+1, dim] (包含SEP token)
            else:
                patch_features = model.gpt.encode(x, apply_mask=False)  # [B, n_patches+1, dim] (包含SEP token)
            
            # GPT的PatchEmbedding会在前面添加一个SEP token，需要去掉以匹配实际的patch数量
            if patch_features.size(1) > x.size(1):  # 如果特征数量大于输入patch数量
                patch_features = patch_features[:, 1:, :]  # 去掉第一个SEP token
            
            return patch_features
        elif isinstance(model, PapageiModel):
            # Papagei模型
            if model == self.teacher_model:
                with torch.no_grad():
                    # 使用Papagei模型的patch特征提取方法
                    patch_features = model.extract_patch_features(x)  # [B, n_patches, dim]
            else:
                patch_features = model.extract_patch_features(x)  # [B, n_patches, dim]
            
            return patch_features
        else:
            return None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.student_model.train()
        total_losses = {
            'total': 0, 'gt': 0, 'pred_distill': 0, 'feature_distill': 0, 'patch_feature_distill': 0, 'patch_distance_distill': 0
        }
        num_batches = 0
        
        # 时间统计
        batch_times = []
        epoch_start_time = time.time()
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f'蒸馏训练 Epoch {epoch+1}')
        
        for batch_idx, data in enumerate(progress_bar):
            batch_start_time = time.time()
            
            ppg_segments = data["ppg_segments"].to(self.device)
            targets = None
            if "ft_label" in data:
                if self.task_type == 'regression':
                    targets = data["ft_label"].to(self.device).float()
                else:
                    targets = data["ft_label"].to(self.device).long()
            
            # Teacher前向传播（无梯度）
            with torch.no_grad():
                teacher_output = self.teacher_model(ppg_segments)
                teacher_features = self.get_features(self.teacher_model, ppg_segments)
                # 获取patch级别特征（如果启用）
                teacher_patch_features = None
                if self.distill_loss.use_patch_feature_distill:
                    teacher_patch_features = self.get_patch_features(self.teacher_model, ppg_segments)
            
            # Student前向传播
            self.optimizer.zero_grad()
            student_output = self.student_model(ppg_segments)
            student_features = self.get_features(self.student_model, ppg_segments)
            # 获取patch级别特征（如果启用）
            student_patch_features = None
            if self.distill_loss.use_patch_feature_distill:
                student_patch_features = self.get_patch_features(self.student_model, ppg_segments)
            
            # 计算蒸馏损失
            loss_dict = self.distill_loss(
                student_output, teacher_output, student_features, 
                teacher_features, targets, self.task_type,
                student_patch_features, teacher_patch_features
            )
            
            # 反向传播
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # 更新学习率调度器（每个batch调用）
            if self.scheduler:
                self.scheduler.step()
            
            # 统计
            for key in total_losses:
                if key == 'total':
                    total_losses[key] += loss_dict['total_loss'].item()
                elif key == 'gt':
                    total_losses[key] += loss_dict['gt_loss'].item()
                elif key == 'pred_distill':
                    total_losses[key] += loss_dict['pred_distill_loss'].item()
                elif key == 'feature_distill':
                    total_losses[key] += loss_dict['feature_distill_loss'].item()
                elif key == 'patch_feature_distill':
                    total_losses[key] += loss_dict['patch_feature_distill_loss'].item()
                elif key == 'patch_distance_distill':
                    total_losses[key] += loss_dict['patch_distance_distill_loss'].item()
            
            num_batches += 1
            
            # 记录batch时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            self.all_train_batch_times.append(batch_time)
            
            # 更新进度条（降低更新频率）
            if batch_idx % 100 == 0 or batch_idx == len(self.train_loader) - 1:  # 每10个batch或最后一个batch更新一次
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else float(self.config['train_config'].get('lr_max', 30e-5))
                progress_bar.set_postfix({
                    'Total': f'{loss_dict["total_loss"].item():.4f}',
                    'GT': f'{loss_dict["gt_loss"].item():.4f}',
                    'Pred': f'{loss_dict["pred_distill_loss"].item():.4f}',
                    'Feat': f'{loss_dict["feature_distill_loss"].item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Batch/s': f'{1.0/batch_time:.2f}'
                })
        
        # 计算平均损失
        avg_losses = {key: total_losses[key] / num_batches for key in total_losses}
        
        # 计算时间统计
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times)
        batches_per_second = 1.0 / avg_batch_time
        
        lprint(f"蒸馏训练统计 - 总时间: {epoch_time:.2f}s, 平均每批: {avg_batch_time:.5f}s, 批次/秒: {batches_per_second:.2f}")
        
        return avg_losses
    
    def validate(self):
        """验证模型"""
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        # 时间统计
        batch_times = []
        epoch_start_time = time.time()
        
        with torch.no_grad():
            for data in tqdm.tqdm(self.val_loader, desc='验证'):
                batch_start_time = time.time()
                
                ppg_segments = data["ppg_segments"].to(self.device)
                
                if "ft_label" in data:
                    if self.task_type == 'regression':
                        labels = data["ft_label"].to(self.device).float()
                    else:
                        labels = data["ft_label"].to(self.device).long()
                    
                    # 只需要student输出进行验证
                    student_output = self.student_model(ppg_segments)
                    
                    if self.task_type == 'regression':
                        loss = F.mse_loss(student_output.squeeze(), labels)
                        all_preds.extend(student_output.squeeze().cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    else:  # classification
                        loss = F.cross_entropy(student_output, labels)
                        all_preds.extend(student_output.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 记录batch时间
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    self.all_val_batch_times.append(batch_time)
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 计算评估指标
        # 确保输入是numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算时间统计
        if batch_times:
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = np.mean(batch_times)
            batches_per_second = 1.0 / avg_batch_time
            lprint(f"验证统计 - 总时间: {epoch_time:.2f}s, 平均每批: {avg_batch_time:.5f}s, 批次/秒: {batches_per_second:.2f}")
        
        if self.task_type == 'regression':
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            return avg_val_loss, {'mse': mse, 'mae': mae}
        else:  # classification
            from sklearn.metrics import accuracy_score, f1_score
            pred_classes = np.argmax(all_preds, axis=1)
            acc = accuracy_score(all_labels, pred_classes)
            f1 = f1_score(all_labels, pred_classes, average='binary')
            return avg_val_loss, {'accuracy': acc, 'f1': f1}
    
    def test(self):
        """测试模型"""
        if self.test_loader is None:
            lprint("警告: 没有测试数据")
            return None, {}
            
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        # 时间统计
        batch_times = []
        epoch_start_time = time.time()
        
        with torch.no_grad():
            for data in tqdm.tqdm(self.test_loader, desc='测试'):
                batch_start_time = time.time()
                
                ppg_segments = data["ppg_segments"].to(self.device)
                
                if "ft_label" in data:
                    if self.task_type == 'regression':
                        labels = data["ft_label"].to(self.device).float()
                    else:
                        labels = data["ft_label"].to(self.device).long()
                    
                    # 只需要student输出进行测试
                    student_output = self.student_model(ppg_segments)
                    
                    if self.task_type == 'regression':
                        loss = F.mse_loss(student_output.squeeze(), labels)
                        all_preds.extend(student_output.squeeze().cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    else:  # classification
                        loss = F.cross_entropy(student_output, labels)
                        all_preds.extend(student_output.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 记录batch时间
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    self.all_test_batch_times.append(batch_time)
        
        avg_test_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 计算评估指标
        # 确保输入是numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算时间统计
        if batch_times:
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = np.mean(batch_times)
            batches_per_second = 1.0 / avg_batch_time
            lprint(f"测试统计 - 总时间: {epoch_time:.2f}s, 平均每批: {avg_batch_time:.5f}s, 批次/秒: {batches_per_second:.2f}")
        
        if self.task_type == 'regression':
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            return avg_test_loss, {'mse': mse, 'mae': mae}
        else:  # classification
            from sklearn.metrics import accuracy_score, f1_score
            pred_classes = np.argmax(all_preds, axis=1)
            acc = accuracy_score(all_labels, pred_classes)
            f1 = f1_score(all_labels, pred_classes, average='binary')
            return avg_test_loss, {'accuracy': acc, 'f1': f1}
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        # 保存学习率调度器状态
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 如果有特征适配器，保存它（全局和patch共用同一个）
        if self.distill_loss.feature_adapter is not None:
            checkpoint['feature_adapter_state_dict'] = self.distill_loss.feature_adapter.state_dict()
        
        # 只保存最佳模型
        if is_best:
            torch.save(checkpoint, f"{self.save_path}_best.pth")
            lprint(f"保存最佳模型: {self.save_path}_best.pth")
    
    def save_training_log(self, log_data):
        """保存训练日志到JSON文件"""
        log_file = f"{self.save_path}_distill_training_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        lprint(f"蒸馏训练日志已保存: {log_file}")
    
    def save_metrics_csv(self, metrics_history):
        """保存训练指标到CSV文件"""
        csv_file = f"{self.save_path}_distill_metrics.csv"
        
        if not metrics_history:
            return
        
        # 获取所有可能的键
        all_keys = set()
        for epoch_data in metrics_history:
            all_keys.update(epoch_data.keys())
        
        # 写入CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(metrics_history)
        
        lprint(f"蒸馏训练指标CSV已保存: {csv_file}")
    
    def train(self):
        """完整训练流程"""
        epochs = self.config['train_config']['epochs']
        save_freq = self.config['train_config'].get('save_freq', 10)
        
        # 初始化日志记录
        training_start_time = datetime.now()
        metrics_history = []
        
        # 获取teacher和student模型参数数量
        teacher_total_params = sum(p.numel() for p in self.teacher_model.parameters())
        teacher_trainable_params = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
        student_total_params = sum(p.numel() for p in self.student_model.parameters())
        student_trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        
        training_log = {
            'experiment_info': {
                'experiment_type': 'knowledge_distillation',
                'teacher_type': 'inferred_from_path',  # 将在main函数中更新
                'student_type': 'inferred_from_config',  # 将在main函数中更新
                'dataset': 'inferred_from_config',  # 将在main函数中更新
                'task_type': self.task_type,
                'start_time': training_start_time.isoformat(),
                'config': self.config,
                'teacher_params': {
                    'total': teacher_total_params,
                    'trainable': teacher_trainable_params
                },
                'student_params': {
                    'total': student_total_params,
                    'trainable': student_trainable_params
                },
                'compression_ratio': teacher_total_params / student_total_params if student_total_params > 0 else 0
            },
            'training_history': [],
            'best_model_info': {}
        }
        
        lprint("开始知识蒸馏训练...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            lprint(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_losses = self.train_epoch(epoch)
            # 显示当前学习率
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else float(self.config['train_config'].get('lr_max', 30e-5))
            lprint(f"当前学习率: {current_lr:.2e}")
            
            lprint(f"训练损失 - 总计: {train_losses['total']:.4f}, "
                   f"GT: {train_losses['gt']:.4f}, "
                   f"预测蒸馏: {train_losses['pred_distill']:.4f}, "
                   f"全局特征蒸馏: {train_losses['feature_distill']:.4f}, "
                   f"patch特征蒸馏: {train_losses['patch_feature_distill']:.4f}, "
                   f"patch距离蒸馏: {train_losses['patch_distance_distill']:.4f}")
            
            # 验证
            val_loss = None
            val_metrics = {}
            if self.val_loader:
                val_loss, val_metrics = self.validate()
                lprint(f"验证损失: {val_loss:.4f}")
                lprint(f"验证指标: {val_metrics}")
                
                # 检查是否为最佳模型
                is_best = val_loss < (self.best_val_loss - self.early_stop_min_delta)
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stop_counter = 0  # 重置早停计数器
                    lprint(f"新的最佳模型! 验证损失: {self.best_val_loss:.4f}")
                else:
                    self.early_stop_counter += 1
                    lprint(f"验证损失未改善 ({self.early_stop_counter}/{self.early_stop_patience})")
            else:
                val_loss = train_losses['total']
                is_best = val_loss < (self.best_val_loss - self.early_stop_min_delta)
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stop_counter = 0  # 重置早停计数器
                else:
                    self.early_stop_counter += 1
                    lprint(f"训练损失未改善 ({self.early_stop_counter}/{self.early_stop_patience})")
            
            # 记录当前epoch的指标
            epoch_time = time.time() - epoch_start_time
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else float(self.config['train_config'].get('lr_max', 30e-5))
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'total_loss': train_losses['total'],
                'gt_loss': train_losses['gt'],
                'pred_distill_loss': train_losses['pred_distill'],
                'feature_distill_loss': train_losses['feature_distill'],
                'patch_feature_distill_loss': train_losses['patch_feature_distill'],
                'patch_distance_distill_loss': train_losses['patch_distance_distill'],
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                'is_best': is_best,
                'early_stop_counter': self.early_stop_counter
            }
            
            # 添加验证指标
            for key, value in val_metrics.items():
                epoch_metrics[f'val_{key}'] = value
            
            metrics_history.append(epoch_metrics)
            
            # 记录到训练日志
            epoch_log = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                'is_best': is_best,
                'timestamp': datetime.now().isoformat()
            }
            training_log['training_history'].append(epoch_log)
            
            # 保存检查点（只保存最佳模型）
            if is_best:
                self.save_checkpoint(epoch, is_best)
                
                # 更新最佳模型信息
                training_log['best_model_info'] = {
                    'epoch': epoch + 1,
                    'val_loss': self.best_val_loss,
                    'train_losses': train_losses,
                    'val_metrics': val_metrics,
                    'model_path': f"{self.save_path}_best.pth"
                }
            
            # 早停检查
            if self.early_stop_counter >= self.early_stop_patience:
                self.early_stopped = True
                lprint(f"🛑 早停触发！连续 {self.early_stop_patience} 个epoch无显著改善")
                lprint(f"最佳验证损失: {self.best_val_loss:.6f} (Epoch {self.best_epoch + 1})")
                break
        
        # 训练结束，更新日志
        training_end_time = datetime.now()
        training_log['experiment_info']['end_time'] = training_end_time.isoformat()
        training_log['experiment_info']['total_training_time'] = str(training_end_time - training_start_time)
        training_log['experiment_info']['best_epoch'] = self.best_epoch + 1
        training_log['experiment_info']['best_val_loss'] = self.best_val_loss
        training_log['experiment_info']['early_stopped'] = self.early_stopped
        training_log['experiment_info']['early_stop_patience'] = self.early_stop_patience
        training_log['experiment_info']['early_stop_min_delta'] = self.early_stop_min_delta
        
        # 计算性能统计
        performance_stats = {}
        if self.all_train_batch_times:
            avg_train_batch_time = np.mean(self.all_train_batch_times)
            performance_stats['avg_train_batch_time_seconds'] = avg_train_batch_time
            performance_stats['avg_train_batches_per_second'] = 1.0 / avg_train_batch_time
            performance_stats['total_train_batches'] = len(self.all_train_batch_times)
        
        if self.all_val_batch_times:
            avg_val_batch_time = np.mean(self.all_val_batch_times)
            performance_stats['avg_val_batch_time_seconds'] = avg_val_batch_time
            performance_stats['avg_val_batches_per_second'] = 1.0 / avg_val_batch_time
            performance_stats['total_val_batches'] = len(self.all_val_batch_times)
        
        training_log['performance_stats'] = performance_stats
        
        if self.early_stopped:
            lprint(f"\n训练提前停止! 最佳验证损失: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
            lprint(f"早停原因: 连续 {self.early_stop_patience} 个epoch验证损失无显著改善 (阈值: {self.early_stop_min_delta})")
        else:
            lprint(f"\n训练完成! 最佳验证损失: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
        
        # 显示性能统计
        if performance_stats:
            lprint("\n蒸馏训练性能统计:")
            if 'avg_train_batches_per_second' in performance_stats:
                lprint(f"平均训练批次/秒: {performance_stats['avg_train_batches_per_second']:.2f} "
                       f"(基于 {performance_stats['total_train_batches']} 个批次)")
            if 'avg_val_batches_per_second' in performance_stats:
                lprint(f"平均验证批次/秒: {performance_stats['avg_val_batches_per_second']:.2f} "
                       f"(基于 {performance_stats['total_val_batches']} 个批次)")
        
        # 保存训练日志和指标CSV
        self.save_training_log(training_log)
        self.save_metrics_csv(metrics_history)
        
        # 训练完成后的测试需要加载最优checkpoint
        if self.test_loader:
            # 加载最佳模型进行测试
            checkpoint_path = f"{self.save_path}_best.pth"
            if os.path.exists(checkpoint_path):
                lprint(f"\n加载最佳模型进行最终测试: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
                
                # 如果有学习率调度器状态，加载它
                if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    lprint("已加载学习率调度器状态")
                
                # 如果有特征适配器，加载它（全局和patch共用同一个）
                if self.distill_loss.feature_adapter is not None and 'feature_adapter_state_dict' in checkpoint:
                    self.distill_loss.feature_adapter.load_state_dict(checkpoint['feature_adapter_state_dict'])
                    lprint("已加载特征适配器")
                
                test_loss, test_metrics = self.test()
                
                # 计算测试性能统计
                test_performance = {}
                if self.all_test_batch_times:
                    avg_test_batch_time = np.mean(self.all_test_batch_times)
                    test_performance['avg_test_batch_time_seconds'] = avg_test_batch_time
                    test_performance['avg_test_batches_per_second'] = 1.0 / avg_test_batch_time
                    test_performance['total_test_batches'] = len(self.all_test_batch_times)
                    
                    lprint(f"蒸馏测试性能统计:")
                    lprint(f"平均测试批次/秒: {test_performance['avg_test_batches_per_second']:.2f} "
                           f"(基于 {test_performance['total_test_batches']} 个批次)")
                
                # 将测试结果添加到日志
                training_log['test_results'] = {
                    'test_loss': test_loss,
                    'test_metrics': test_metrics,
                    'test_time': datetime.now().isoformat(),
                    'test_performance': test_performance
                }
                
                # 重新保存包含测试结果的日志
                self.save_training_log(training_log)
                
                return test_loss, test_metrics
            else:
                lprint(f"警告: 未找到最佳模型文件 {checkpoint_path}，跳过测试")
                training_log['test_results'] = {
                    'error': 'Best model file not found',
                    'test_time': datetime.now().isoformat()
                }
                self.save_training_log(training_log)
                return None, {}
        else:
            training_log['test_results'] = {
                'warning': 'No test data provided',
                'test_time': datetime.now().isoformat()
            }
            self.save_training_log(training_log)
            return None, {}


def set_seed(seed):
    """设置随机种子确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """加载配置文件（支持YAML和JSON）"""
    assert os.path.isfile(config_path), f'{config_path=}'
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")
    
    lprint(f"配置文件已加载: {config_path}")
    lprint(json.dumps(config, indent=2))
    return config


def auto_load_distill_configs(teacher_type, student_type, dataset):
    """根据teacher、student类型和数据集自动加载蒸馏配置文件"""
    # 推断蒸馏配置文件路径
    if teacher_type == 'papagei' and student_type == 'mlp':
        distill_config_path = "config/distillation/papagei_to_mlp_distill.yaml"
    elif teacher_type == 'papagei' and student_type == 'linear':
        distill_config_path = "config/distillation/papagei_to_linear_distill.yaml"
    elif teacher_type == 'papagei' and student_type.startswith('gpt'):
        # Papagei到GPT的蒸馏，支持patch级别特征蒸馏
        distill_config_path = "config/distillation/papagei_to_gpt_distill.yaml"
    elif teacher_type.startswith('gpt') and student_type.startswith('gpt'):
        # GPT到GPT的蒸馏，支持patch级别特征蒸馏
        distill_config_path = "config/distillation/gpt_to_gpt_patch_distill.yaml"
        # 如果不存在patch配置，回退到普通配置
        if not os.path.exists(distill_config_path):
            distill_config_path = "config/distillation/gpt_to_mlp_distill.yaml"
            lprint(f"警告: 未找到GPT到GPT的patch蒸馏配置，使用默认配置")
    elif teacher_type.startswith('gpt') and student_type == 'mlp':
        distill_config_path = "config/distillation/gpt_to_mlp_distill.yaml"
    elif teacher_type.startswith('gpt') and student_type == 'linear':
        distill_config_path = "config/distillation/gpt_to_linear_distill.yaml"
    elif teacher_type == 'mlp' and student_type == 'linear':
        distill_config_path = "config/distillation/mlp_to_linear_distill.yaml"
    else:
        # 默认使用gpt_to_mlp配置
        distill_config_path = "config/distillation/gpt_to_mlp_distill.yaml"
    
    # 推断学生模型配置文件路径
    if student_type in ['gpt_19m', 'gpt_1m']:
        student_config_path = f"config/models/gpt_config_{dataset}.yaml"
    else:
        student_config_path = f"config/models/{student_type}_config_{dataset}.yaml"
    
    # 数据配置文件路径
    data_config_path = f"config/data/{dataset}_data.yaml"
    
    # 加载配置
    distill_config = load_config(distill_config_path)
    student_model_config = load_config(student_config_path)
    data_config = load_config(data_config_path)
    
    # 合并配置
    config = {
        'distillation': distill_config['distillation'],
        'train_config': distill_config['train_config'],
        'data_config': data_config['data_config']
    }
    
    # 获取任务类型
    task_type = data_config['data_config'].get('task_type', 'regression')
    
    return config, student_model_config['model_config'], task_type


def auto_infer_teacher_path(teacher_type, dataset, save_dir):
    """自动推断teacher模型路径"""
    teacher_path = os.path.join(save_dir, f"{teacher_type}_{dataset}_best.pth")
    return teacher_path


def create_model(model_config, model_type='gpt_19m'):
    """创建指定类型的模型"""
    if model_type.lower() == 'papagei':
        try:
            model = create_papagei_model(model_config)
        except Exception as e:
            lprint(f"创建Papagei模型失败: {e}")
            model = PapageiModel(model_config)
    elif model_type.lower() in ['gpt', 'gpt_19m', 'gpt_1m']:
        # 如果是GPT模型，根据类型加载对应配置
        if model_type.lower() == 'gpt_1m':
            # 加载GPT-1M配置
            gpt_1m_config_path = 'config/gpt_1M.json'
            if os.path.exists(gpt_1m_config_path):
                gpt_1m_config = load_config(gpt_1m_config_path)
                # 合并配置，优先使用GPT-1M的架构参数
                merged_config = model_config.copy()
                merged_config.update(gpt_1m_config)
            else:
                # 使用默认1M配置
                gpt_1m_config = {
                    'd_model': 128,
                    'n_heads': 4,
                    'n_layers': 5,
                    'patch_size': 40,
                    'dropout': 0.1,
                    'max_len': 2400,
                    'PARAMS': '1M'
                }
                merged_config = model_config.copy()
                merged_config.update(gpt_1m_config)
            
            # 添加缺失的必需字段
            required_fields = {
                'tune_mode': 'full',
                'use_penultimate_layer': False,
                'is_input_multichannel': False,
                'ecg_input': False,
                'fuse_tuning': False,
                'fuse_feat_type': None,
                'use_lora': False,
                'out_dim_override': None,
                'pooling_fxn': 'linear'
            }
            
            for key, default_value in required_fields.items():
                if key not in merged_config:
                    merged_config[key] = default_value
            
            # 确保PARAMS字段正确设置
            merged_config['PARAMS'] = '1M'
            model = GPT_with_linearOutput(merged_config)
        else:
            # GPT-19M或默认GPT
            model = GPT_with_linearOutput(model_config)
    elif model_type.lower() == 'linear':
        try:
            model = create_linear_model(model_config)
        except:
            model = LinearModel(model_config)
    elif model_type.lower() == 'mlp':
        try:
            model = create_mlp_model(model_config)
        except:
            # 计算输入大小
            patch_size = model_config.get('patch_size', 40)
            n_patches = model_config.get('n_patches', 30)
            input_size = patch_size * n_patches
            
            mlp_config = {
                'input_size': input_size,
                'hidden_sizes': model_config.get('hidden_sizes', [512, 256, 128]),
                'output_size': model_config.get('output_size', 1),
                'dropout': model_config.get('dropout', 0.2),
                'activation': model_config.get('activation', 'relu'),
                'batch_norm': model_config.get('batch_norm', True)
            }
            model = MLP(**mlp_config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    lprint(f"创建{model_type.upper()}模型:")
    lprint(model)
    return model


def load_teacher_model(teacher_path, teacher_type, device, n_patches=None):
    """加载预训练的teacher模型"""
    lprint(f"加载teacher模型: {teacher_path} (类型: {teacher_type})")
    
    checkpoint = torch.load(teacher_path, map_location='cpu')
    
    # 推断数据集（从teacher路径）
    if 'dalia' in teacher_path:
        dataset = 'dalia'
    elif 'stanfordAF' in teacher_path:
        dataset = 'stanfordAF'
    else:
        dataset = 'dalia'  # 默认使用dalia
    
    # 加载完整的模型配置
    if teacher_type in ['gpt_19m', 'gpt_1m']:
        model_config_path = f"config/models/gpt_config_{dataset}.yaml"
    else:
        model_config_path = f"config/models/{teacher_type}_config_{dataset}.yaml"
    
    if os.path.exists(model_config_path):
        full_config = load_config(model_config_path)
        teacher_config = full_config['model_config'].copy()
        
        # 添加缺失的必需字段（仅对GPT模型）
        if teacher_type in ['gpt_19m', 'gpt_1m']:
            required_fields = {
                'gpt_state_dict_path': None,
                'strict_loading_gpt_state_dict': True,
                'tune_mode': 'full',
                'use_penultimate_layer': False,
                'is_input_multichannel': False,
                'ecg_input': False,
                'fuse_tuning': False,
                'fuse_feat_type': None,
                'use_lora': False,
                'out_dim_override': 2,
                'pooling_fxn': 'linear'
            }
            
            for key, default_value in required_fields.items():
                if key not in teacher_config:
                    teacher_config[key] = default_value
        
        # 如果提供了n_patches，使用计算得到的值
        if n_patches is not None:
            teacher_config['n_patches'] = n_patches
        
        # 根据模型类型调整架构参数
        if teacher_type == 'gpt_1m':
            # 加载GPT-1M的架构参数
            gpt_1m_config_path = 'config/gpt_1M.json'
            if os.path.exists(gpt_1m_config_path):
                gpt_1m_config = load_config(gpt_1m_config_path)
                # 更新架构相关参数
                for key in ['d_model', 'n_heads', 'n_layers', 'dropout', 'max_len']:
                    if key in gpt_1m_config:
                        teacher_config[key] = gpt_1m_config[key]
                teacher_config['PARAMS'] = '1M'
        elif teacher_type == 'gpt_19m':
            # 确保使用GPT-19M的参数
            teacher_config['PARAMS'] = '19M'
    else:
        # 如果找不到配置文件，使用默认配置
        lprint(f"警告: 未找到配置文件 {model_config_path}，使用默认配置")
        teacher_config = {
            'model_name': 'gpt',
            'PARAMS': '1M' if teacher_type == 'gpt_1m' else '19M',
            'patch_size': 40,
            'd_model': 128 if teacher_type == 'gpt_1m' else 512,
            'n_heads': 4 if teacher_type == 'gpt_1m' else 8,
            'n_layers': 5 if teacher_type == 'gpt_1m' else 6,
            'dropout': 0.1,
            'max_len': 2400,
            'loss': 'laplace',
            'with_conv': False,
            'output_size': 1,
            'out_classes': 1,
            'apply_mask': False,
            'n_patches': n_patches if n_patches is not None else 30,
            'gpt_state_dict_path': None,
            'strict_loading_gpt_state_dict': True,
            'tune_mode': 'full',
            'use_penultimate_layer': False,
            'is_input_multichannel': False,
            'ecg_input': False,
            'fuse_tuning': False,
            'fuse_feat_type': None,
            'use_lora': False,
            'out_dim_override': 2,
            'pooling_fxn': 'linear'
        }
    
    # 创建teacher模型
    teacher_model = create_model(teacher_config, teacher_type)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'student_model_state_dict' in checkpoint:
        state_dict = checkpoint['student_model_state_dict']
    elif 'model' in checkpoint:
        # 这是从train.py保存的checkpoint格式
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除torch.compile产生的_orig_mod.前缀
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]  # 移除前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    teacher_model.load_state_dict(state_dict)
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 设为评估模式
    
    # 冻结teacher模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    lprint("Teacher模型加载完成并已冻结")
    return teacher_model


def count_parameters(model):
    """计算模型参数数量"""
    # 如果是Papagei模型，使用专门的参数计算函数
    if isinstance(model, PapageiModel):
        return count_papagei_parameters(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lprint(f"总参数数: {total_params:,}")
    lprint(f"可训练参数数: {trainable_params:,}")
    return total_params, trainable_params


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='统一PPG模型知识蒸馏训练')
    parser.add_argument('--teacher_type', type=str, required=True,
                      choices=['gpt_19m', 'gpt_1m', 'linear', 'mlp', 'papagei'], help='Teacher模型类型')
    parser.add_argument('--student_type', type=str, required=True,
                      choices=['gpt_19m', 'gpt_1m', 'linear', 'mlp', 'papagei'], help='Student模型类型')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['dalia', 'stanfordAF'], help='数据集名称')
    parser.add_argument('--teacher_path', type=str, help='Teacher模型路径 (可选，自动推断)')
    parser.add_argument('--save_dir', type=str, default='./output', help='Teacher模型保存目录')
    parser.add_argument('--save_dir_student', type=str, default='./output_s', help='Student模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_only', action='store_true', help='只进行测试')
    parser.add_argument('--no-test', action='store_true', help='禁用训练完成后的自动测试')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 自动加载配置文件
    config, student_config, task_type = auto_load_distill_configs(
        args.teacher_type, args.student_type, args.dataset
    )
    
    # 自动推断teacher模型路径（如果未提供）
    if args.teacher_path is None:
        teacher_path = auto_infer_teacher_path(args.teacher_type, args.dataset, args.save_dir)
    else:
        teacher_path = args.teacher_path
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprint(f"使用设备: {device}")
    lprint(f"数据集: {args.dataset}, Teacher: {args.teacher_type}, Student: {args.student_type}, 任务类型: {task_type}")
    
    # 创建保存目录
    os.makedirs(args.save_dir_student, exist_ok=True)
    
    # 构建保存路径
    save_path = os.path.join(args.save_dir_student, 
                           f"distill_{args.teacher_type}_to_{args.student_type}_{args.dataset}")
    
    # 数据加载和n_patches计算
    data_cfg = config['data_config']
    lprint("加载数据...")
    
    # 获取patch_size
    patch_size = student_config.get('patch_size', 40)
    
    # 计算n_patches
    n_patches = calc_n_patches(data_cfg['train_data_path'], patch_size)
    lprint(f"计算得到的patch数量: {n_patches}")
    
    # 更新student_config中的n_patches
    student_config['n_patches'] = n_patches
    
    # 加载teacher模型
    teacher_model = load_teacher_model(teacher_path, args.teacher_type, device, n_patches)
    lprint("Teacher模型:")
    count_parameters(teacher_model)
    
    # 创建student模型
    student_model = create_model(student_config, args.student_type)
    student_model = student_model.to(device)
    lprint("Student模型:")
    count_parameters(student_model)
    
    train_dataset = PretrainDataset(
        data_cfg['train_data_path'],
        patch_size=patch_size,
        train_labels_dataset_path=data_cfg.get('train_label_path', ''),
        data_red_factor=data_cfg.get('data_red_factor', 1)
    )
    
    val_dataset = None
    if data_cfg.get('val_data_path'):
        val_dataset = PretrainDataset(
            data_cfg['val_data_path'],
            patch_size=patch_size,
            train_labels_dataset_path=data_cfg.get('val_label_path', ''),
            data_red_factor=data_cfg.get('data_red_factor', 1)
        )
    
    # 测试数据集
    test_dataset = None
    if data_cfg.get('test_data_path'):
        test_dataset = PretrainDataset(
            data_cfg['test_data_path'],
            patch_size=patch_size,
            train_labels_dataset_path=data_cfg.get('test_label_path', ''),
            data_red_factor=data_cfg.get('data_red_factor', 1)
        )
    
    train_config = config['train_config']
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    # 如果只是测试，加载模型并测试
    if args.test_only:
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            student_model.load_state_dict(checkpoint['student_model_state_dict'])
            lprint(f"已加载学生模型: {checkpoint_path}")
            
            if test_loader:
                trainer = DistillationTrainer(
                    teacher_model=teacher_model,
                    student_model=student_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    device=device,
                    save_path=save_path,
                    task_type=task_type,
                    test_loader=test_loader
                )
                
                # 如果有学习率调度器状态，加载它
                if trainer.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    lprint("已加载学习率调度器状态")
                
                # 如果有特征适配器，加载它（全局和patch共用同一个）
                if trainer.distill_loss.feature_adapter is not None and 'feature_adapter_state_dict' in checkpoint:
                    trainer.distill_loss.feature_adapter.load_state_dict(checkpoint['feature_adapter_state_dict'])
                    lprint("已加载特征适配器")
                
                test_loss, test_metrics = trainer.test()
                lprint(f"测试损失: {test_loss:.4f}")
                lprint(f"测试指标: {test_metrics}")
            else:
                lprint("未找到测试数据")
        else:
            lprint(f"未找到模型文件: {checkpoint_path}")
        return
    
    # 创建蒸馏训练器
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_path=save_path,
        task_type=task_type,
        test_loader=test_loader
    )
    
    # 在训练开始前更新trainer的日志信息（通过monkey patch）
    def update_training_log_info(trainer, teacher_type, student_type, dataset):
        """更新训练器的日志信息"""
        # 这个方法会在train()方法中被调用来更新experiment_info
        original_train = trainer.train
        
        def enhanced_train():
            result = original_train()
            # 训练完成后，读取并更新日志文件
            try:
                log_file = f"{trainer.save_path}_distill_training_log.json"
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        training_log = json.load(f)
                    
                    # 更新实验信息
                    training_log['experiment_info']['teacher_type'] = teacher_type
                    training_log['experiment_info']['student_type'] = student_type
                    training_log['experiment_info']['dataset'] = dataset
                    training_log['experiment_info']['teacher_path'] = teacher_path
                    
                    # 重新保存
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(training_log, f, indent=2, ensure_ascii=False)
            except Exception as e:
                lprint(f"更新日志信息时出错: {e}")
            
            return result
        
        trainer.train = enhanced_train
    
    # 应用日志信息更新
    update_training_log_info(trainer, args.teacher_type, args.student_type, args.dataset)
    
    # 开始训练
    test_loss, test_metrics = trainer.train()
    
    # 如果启用了测试且有测试数据，显示最终测试结果
    if not args.no_test and test_loader and test_loss is not None:
        lprint(f"\n最终测试损失: {test_loss:.4f}")
        lprint(f"最终测试指标: {test_metrics}")
    elif not args.no_test and not test_loader:
        lprint("警告: 启用了自动测试但未找到测试数据")


if __name__ == "__main__":
    main() 