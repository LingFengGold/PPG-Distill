#!/usr/bin/env python

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

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    
    if not squared:
        res = res.sqrt()
    
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
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

from model.gpt import GPT_with_linearOutput
from model.linear import LinearModel, create_linear_model
from model.mlp import MLP, create_mlp_model
from model.papagei import PapageiModel, create_papagei_model, count_papagei_parameters
from data.pretrain_dataset import PretrainDataset
try:
    from local.supp_fxns import *
except ImportError:
    pass  


def calc_n_patches(filepath, fs):   
    assert os.path.isfile(filepath), f'{filepath=}'
    seq_length = get_numpy_array_metadata(filepath, return_attrs=True)[0][-1]
    assert seq_length >= fs, 'need atleast 1 second signal length'
    assert seq_length % fs == 0, 'signal length (in seconds) must be a whole number'
    n_patches = seq_length // fs
    return n_patches


class DistillationLoss(nn.Module):  
    
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, temperature=4.0, 
                 use_feature_distill=True, feature_loss_weight=1.0,
                 use_patch_feature_distill=False, patch_feature_loss_weight=1.0,
                 patch_distill_mode='direct', contrastive_temperature=0.1,
                 use_patch_distance_distill=False, patch_distance_loss_weight=1.0,
                 tie_feature_and_patch=True):    
        """
        Args:
            alpha: Prediction distillation loss weight (overrides config file)
            beta: Global feature distillation loss weight (overrides config file)
            gamma: Patch-level distillation loss weight (overrides config file)
            temperature: Softening temperature
            use_feature_distill: Whether to use global feature distillation
            feature_loss_weight: Global feature distillation loss weight
            use_patch_feature_distill: Whether to use patch-level feature distillation
            patch_feature_loss_weight: Patch feature distillation loss weight
            patch_distill_mode: Patch distillation mode ('direct' or 'contrastive')
            contrastive_temperature: Contrastive learning temperature parameter (or 'auto')
            use_patch_distance_distill: Whether to use patch distance distillation
            patch_distance_loss_weight: Patch distance distillation loss weight
            tie_feature_and_patch: Whether to merge feature and patch representations
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
        
        self.tie_feature_and_patch = tie_feature_and_patch
        
        self.feature_adapter = None
        
        self.patch_distance_distiller = RkdDistance() if use_patch_distance_distill else None
        
    def setup_feature_adapter(self, teacher_dim, student_dim, teacher_patch_dim=None, student_patch_dim=None):
        
        if teacher_patch_dim is not None and teacher_patch_dim != teacher_dim:
            lprint(f"Warning: teacher global feature dimension ({teacher_dim}) does not match patch feature dimension ({teacher_patch_dim})")
        
        if student_patch_dim is not None and student_patch_dim != student_dim:
            lprint(f"Warning: student global feature dimension ({student_dim}) does not match patch feature dimension ({student_patch_dim})")
        
        need_adapter = (
            (self.use_feature_distill and teacher_dim != student_dim) or
            (self.use_patch_feature_distill and teacher_dim != student_dim)
        )
        
        if need_adapter:
            self.feature_adapter = nn.Linear(teacher_dim, student_dim)
            lprint(f"Created shared feature adapter: {teacher_dim} -> {student_dim} (global and patch features shared)")
        else:
            lprint("Feature dimension matches, no adapter needed")
    

        
    def forward(self, student_output, teacher_output, student_features=None, 
                teacher_features=None, targets=None, task_type='regression',
                student_patch_features=None, teacher_patch_features=None):
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

        global_feat_loss = None
        if self.use_feature_distill and student_features is not None and teacher_features is not None:
            if len(teacher_features.shape) == 3:
                teacher_features = teacher_features.mean(dim=1)
            teacher_feat_adapted = self.feature_adapter(teacher_features) if self.feature_adapter is not None else teacher_features
            global_feat_loss = F.mse_loss(student_features, teacher_feat_adapted)

        patch_feat_loss = None
        if self.use_patch_feature_distill and student_patch_features is not None and teacher_patch_features is not None:
            if self.feature_adapter is not None:
                teacher_patch_adapted = self.feature_adapter(teacher_patch_features)
            else:
                teacher_patch_adapted = teacher_patch_features
            
            if self.patch_distill_mode == 'direct':
                patch_feat_loss = F.mse_loss(student_patch_features, teacher_patch_adapted)
            else:
                patch_feat_loss = self._compute_contrastive_patch_loss(student_patch_features, teacher_patch_adapted)

        patch_distance_loss = None
        if self.use_patch_distance_distill and student_patch_features is not None and teacher_patch_features is not None:
            batch_size = student_patch_features.size(0)
            patch_distance_losses = []
            
            for b in range(batch_size):
                student_patches_b = student_patch_features[b]  # [n_patches, dim]
                teacher_patches_b = teacher_patch_features[b]  # [n_patches, dim]
                
                patch_dist_loss_b = self.patch_distance_distiller(student_patches_b, teacher_patches_b)
                patch_distance_losses.append(patch_dist_loss_b)
            
            patch_distance_loss = torch.stack(patch_distance_losses).mean()

        ngt = gt_loss
        npred = pred_loss
        nfeat = global_feat_loss
        npatch = patch_feat_loss
        npatch_distance = patch_distance_loss

        npatch_combined = None
        if self.tie_feature_and_patch:
            patch_parts = [x for x in [npatch, npatch_distance] if x is not None]
            npatch_combined = sum(patch_parts) if patch_parts else None

        loss_dict = {}
        
        total = 0.0
        if ngt is not None:
            total += 0.0 * ngt  
        total += self.alpha * npred  
        
        if self.tie_feature_and_patch:
            if nfeat is not None:
                total += self.beta * nfeat  
            if npatch_combined is not None:
                total += self.gamma * npatch_combined  
        else:
            if nfeat is not None:
                total += self.beta * nfeat  
            if npatch is not None:
                total += self.gamma * npatch  
            if npatch_distance is not None:
                total += self.gamma * npatch_distance  

        loss_dict['gt_loss'] = ngt if ngt is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['pred_distill_loss'] = npred
        loss_dict['feature_distill_loss'] = nfeat if nfeat is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['patch_feature_distill_loss'] = npatch if npatch is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['patch_distance_distill_loss'] = npatch_distance if npatch_distance is not None else torch.tensor(0.0, device=npred.device)
        loss_dict['total_loss'] = total
        return loss_dict
    
    def _compute_patch_distillation_loss(self, student_patch_features, teacher_patch_features):
        if self.feature_adapter is not None:
            B, n_patches, teacher_dim = teacher_patch_features.shape
            teacher_patch_flat = teacher_patch_features.view(B * n_patches, teacher_dim)
            teacher_patch_adapted_flat = self.feature_adapter(teacher_patch_flat)
            student_dim = teacher_patch_adapted_flat.shape[-1]
            teacher_patch_features_adapted = teacher_patch_adapted_flat.view(B, n_patches, student_dim)
        else:
            teacher_patch_features_adapted = teacher_patch_features
        
        if self.patch_distill_mode == 'direct':
            patch_distill_loss = F.mse_loss(student_patch_features, teacher_patch_features_adapted)
            
        elif self.patch_distill_mode == 'contrastive':
            patch_distill_loss = self._compute_contrastive_patch_loss(
                student_patch_features, teacher_patch_features_adapted
            )
        else:
            raise ValueError(f"Unsupported patch distillation mode: {self.patch_distill_mode}")
        
        return patch_distill_loss
    
    def _compute_contrastive_patch_loss(self, student_patches, teacher_patches):
        batch_size, n_patches, dim = student_patches.shape
        
        if isinstance(self.contrastive_temperature, str) and self.contrastive_temperature == 'auto':
            tau = 1.0 / math.sqrt(dim)
        else:
            tau = self.contrastive_temperature
        
        student_patches_norm = F.normalize(student_patches, dim=-1)  # [B, n_patches, dim]
        teacher_patches_norm = F.normalize(teacher_patches, dim=-1)  # [B, n_patches, dim]
        
        total_loss = 0.0
        
        for b in range(batch_size):
            student_b = student_patches_norm[b]  # [n_patches, dim]
            teacher_b = teacher_patches_norm[b]   # [n_patches, dim]
            
            # similarities: [n_patches, n_patches]
            similarities = torch.mm(student_b, teacher_b.t()) / tau
            
            labels = torch.arange(n_patches, device=similarities.device)
            
            contrastive_loss_b = F.cross_entropy(similarities, labels)
            total_loss += contrastive_loss_b
        
        return total_loss / batch_size


class DistillationTrainer:
    
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
            tie_feature_and_patch=distill_config.get('tie_feature_and_patch', True)
        )
        
        teacher_dim = self.get_feature_dim(teacher_model)
        student_dim = self.get_feature_dim(student_model)
        
        teacher_patch_dim = None
        student_patch_dim = None
        if self.distill_loss.use_patch_feature_distill:
            teacher_patch_dim = self.get_patch_feature_dim(teacher_model)
            student_patch_dim = self.get_patch_feature_dim(student_model)
        
        self.distill_loss.setup_feature_adapter(teacher_dim, student_dim, teacher_patch_dim, student_patch_dim)
        
        if self.distill_loss.feature_adapter is not None:
            self.distill_loss.feature_adapter = self.distill_loss.feature_adapter.to(device)
            lprint(f"Feature adapter moved to device: {device}")
        
        if self.distill_loss.patch_distance_distiller is not None:
            self.distill_loss.patch_distance_distiller = self.distill_loss.patch_distance_distiller.to(device)
            lprint(f"Patch distance distiller moved to device: {device}")
        
        optimizer_params = list(self.student_model.parameters())
        
        if self.distill_loss.feature_adapter is not None:
            optimizer_params.extend(self.distill_loss.feature_adapter.parameters())
            lprint("Feature adapter parameters added to optimizer")
        
        train_config = config['train_config']
        optimizer_type = train_config.get('optimizer', 'Adam').lower()
        lr = float(train_config.get('lr_init', 1e-5))
        weight_decay = float(train_config.get('weight_decay', 0.0))
        
        if optimizer_type == 'adam':
            self.optimizer = Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        self.scheduler = None
        
        lr_init = float(train_config.get('lr_init', 1e-5))
        lr_max = float(train_config.get('lr_max', 30e-5))
        lr_final = float(train_config.get('lr_final', 1e-6))
        lr_schedule_ratio = float(train_config.get('lr_schedule_ratio', 1))
        lr_warm_up = float(train_config.get('lr_warm_up', 0.25))
        
        epochs = train_config['epochs']
        batches_per_epoch = len(train_loader)
        lr_schedule_step = int(lr_schedule_ratio * epochs * batches_per_epoch)
        warm_up_steps = int(lr_schedule_step * lr_warm_up)

        lprint(f"  lr_init: {lr_init}, lr_max: {lr_max}, lr_final: {lr_final}")
        lprint(f"  Total schedule steps: {lr_schedule_step}, Warmup steps: {warm_up_steps}")
        lprint(f"  Batches per epoch: {batches_per_epoch}")
        
        lambda_warmup = lambda step: (lr_init + step * (lr_max - lr_init) / warm_up_steps) / lr_init
        warm_up_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_warmup)
        
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=lr_schedule_step - warm_up_steps, eta_min=lr_final)
        
        self.scheduler = LrScheduler(self.optimizer, warm_up_scheduler, main_scheduler, warm_up_steps, lr_schedule_step)
        
        lprint(f"Created batch-level learning rate scheduler (warmup + cosine annealing)")
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        self.early_stop_patience = train_config.get('early_stop_patience', 20)  
        self.early_stop_min_delta = float(train_config.get('early_stop_min_delta', 1e-6))  
        self.early_stop_counter = 0
        self.early_stopped = False
        
        self.all_train_batch_times = []
        self.all_val_batch_times = []
        self.all_test_batch_times = []
        
    def get_feature_dim(self, model):
        model.eval()
        with torch.no_grad():
            probe = next(iter(self.train_loader))["ppg_segments"].to(self.device)[:2]
            features = self.get_features(model, probe)
            
            if features.dim() == 3:
                feature_dim = features.shape[-1]
            else:
                feature_dim = features.shape[-1]
            
            lprint(f"Detected model global feature dimension: {feature_dim}")
            return feature_dim
    
    def get_patch_feature_dim(self, model):
        if hasattr(model, 'gpt'):
            model.eval()
            with torch.no_grad():
                probe = next(iter(self.train_loader))["ppg_segments"].to(self.device)[:2]
                patch_features = self.get_patch_features(model, probe)
                
                if patch_features is None:
                    return None
                
                patch_feature_dim = patch_features.shape[-1]  # [B, n_patches, dim]
                lprint(f"Detected GPT model patch feature dimension: {patch_feature_dim}")
                return patch_feature_dim
        elif isinstance(model, PapageiModel):
            model.eval()
            with torch.no_grad():
                probe = next(iter(self.train_loader))["ppg_segments"].to(self.device)[:2]
                patch_features = self.get_patch_features(model, probe)
                
                if patch_features is None:
                    return None
                
                patch_feature_dim = patch_features.shape[-1]  # [B, n_patches, dim]
                lprint(f"Detected Papagei model patch feature dimension: {patch_feature_dim}")
                return patch_feature_dim
        else:
            lprint(f"Model {type(model).__name__} does not support patch-level feature distillation")
            return None
    
    def get_features(self, model, x):
        if isinstance(model, PapageiModel):
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
        if hasattr(model, 'gpt'):
            if model == self.teacher_model:
                with torch.no_grad():
                    patch_features = model.gpt.encode(x, apply_mask=False)  # [B, n_patches+1, dim] (contains SEP token)
            else:
                patch_features = model.gpt.encode(x, apply_mask=False)  # [B, n_patches+1, dim] (contains SEP token)
            
            if patch_features.size(1) > x.size(1):  # if feature number is greater than input patch number
                patch_features = patch_features[:, 1:, :]  # remove the first SEP token
            
            return patch_features
        elif isinstance(model, PapageiModel):
            if model == self.teacher_model:
                with torch.no_grad():
                    patch_features = model.extract_patch_features(x)  # [B, n_patches, dim]
            else:
                patch_features = model.extract_patch_features(x)  # [B, n_patches, dim]
            
            return patch_features
        else:
            return None
    
    def train_epoch(self, epoch):
        self.student_model.train()
        total_losses = {
            'total': 0, 'gt': 0, 'pred_distill': 0, 'feature_distill': 0, 'patch_feature_distill': 0, 'patch_distance_distill': 0
        }
        num_batches = 0
        
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
            
            with torch.no_grad():
                teacher_output = self.teacher_model(ppg_segments)
                teacher_features = self.get_features(self.teacher_model, ppg_segments)
                teacher_patch_features = None
                if self.distill_loss.use_patch_feature_distill:
                    teacher_patch_features = self.get_patch_features(self.teacher_model, ppg_segments)
            
            self.optimizer.zero_grad()
            student_output = self.student_model(ppg_segments)
            student_features = self.get_features(self.student_model, ppg_segments)
            student_patch_features = None
            if self.distill_loss.use_patch_feature_distill:
                student_patch_features = self.get_patch_features(self.student_model, ppg_segments)
            
            loss_dict = self.distill_loss(
                student_output, teacher_output, student_features, 
                teacher_features, targets, self.task_type,
                student_patch_features, teacher_patch_features
            )
            
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
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
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            self.all_train_batch_times.append(batch_time)
            
            if batch_idx % 100 == 0 or batch_idx == len(self.train_loader) - 1:  
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else float(self.config['train_config'].get('lr_max', 30e-5))
                progress_bar.set_postfix({
                    'Total': f'{loss_dict["total_loss"].item():.4f}',
                    'GT': f'{loss_dict["gt_loss"].item():.4f}',
                    'Pred': f'{loss_dict["pred_distill_loss"].item():.4f}',
                    'Feat': f'{loss_dict["feature_distill_loss"].item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Batch/s': f'{1.0/batch_time:.2f}'
                })
        
        avg_losses = {key: total_losses[key] / num_batches for key in total_losses}
        
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times)
        batches_per_second = 1.0 / avg_batch_time
        
        lprint(f"Distillation training statistics - Total time: {epoch_time:.2f}s, Average batch time: {avg_batch_time:.5f}s, Batches per second: {batches_per_second:.2f}")
        
        return avg_losses
    
    def validate(self):
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
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
                    
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    self.all_val_batch_times.append(batch_time)
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if batch_times:
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = np.mean(batch_times)
            batches_per_second = 1.0 / avg_batch_time
            lprint(f"Validation statistics - Total time: {epoch_time:.2f}s, Average batch time: {avg_batch_time:.5f}s, Batches per second: {batches_per_second:.2f}")
        
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
        if self.test_loader is None:
            lprint("Warning: No test data")
            return None, {}
            
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
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
                    
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    self.all_test_batch_times.append(batch_time)
        
        avg_test_loss = total_loss / num_batches if num_batches > 0 else 0
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if batch_times:
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = np.mean(batch_times)
            batches_per_second = 1.0 / avg_batch_time
            lprint(f"Test statistics - Total time: {epoch_time:.2f}s, Average batch time: {avg_batch_time:.5f}s, Batches per second: {batches_per_second:.2f}")
        
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
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.distill_loss.feature_adapter is not None:
            checkpoint['feature_adapter_state_dict'] = self.distill_loss.feature_adapter.state_dict()
        
        if is_best:
            torch.save(checkpoint, f"{self.save_path}_best.pth")
            lprint(f"Saved best model: {self.save_path}_best.pth")
    
    def save_training_log(self, log_data):
        log_file = f"{self.save_path}_distill_training_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        lprint(f"Distillation training log saved: {log_file}")
    
    def save_metrics_csv(self, metrics_history):
        csv_file = f"{self.save_path}_distill_metrics.csv"
        
        if not metrics_history:
            return
        
        all_keys = set()
        for epoch_data in metrics_history:
            all_keys.update(epoch_data.keys())
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(metrics_history)
        
        lprint(f"Distillation training metrics CSV saved: {csv_file}")
    
    def train(self):
        epochs = self.config['train_config']['epochs']
        save_freq = self.config['train_config'].get('save_freq', 10)
        
        training_start_time = datetime.now()
        metrics_history = []
        
        teacher_total_params = sum(p.numel() for p in self.teacher_model.parameters())
        teacher_trainable_params = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
        student_total_params = sum(p.numel() for p in self.student_model.parameters())
        student_trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        
        training_log = {
            'experiment_info': {
                'experiment_type': 'knowledge_distillation',
                'teacher_type': 'inferred_from_path',  
                'student_type': 'inferred_from_config',  
                'dataset': 'inferred_from_config',  
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
        
        lprint("Start knowledge distillation training...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            lprint(f"\nEpoch {epoch+1}/{epochs}")
            
            train_losses = self.train_epoch(epoch)
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else float(self.config['train_config'].get('lr_max', 30e-5))
            lprint(f"Current learning rate: {current_lr:.2e}")
            
            lprint(f"Training loss - Total: {train_losses['total']:.4f}, "
                   f"GT: {train_losses['gt']:.4f}, "
                   f"Predict distillation: {train_losses['pred_distill']:.4f}, "
                   f"Global feature distillation: {train_losses['feature_distill']:.4f}, "
                   f"Patch feature distillation: {train_losses['patch_feature_distill']:.4f}, "
                   f"Patch distance distillation: {train_losses['patch_distance_distill']:.4f}")
            
            val_loss = None
            val_metrics = {}
            if self.val_loader:
                val_loss, val_metrics = self.validate()
                lprint(f"Validation loss: {val_loss:.4f}")
                lprint(f"Validation metrics: {val_metrics}")
                
                is_best = val_loss < (self.best_val_loss - self.early_stop_min_delta)
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stop_counter = 0  
                    lprint(f"New best model! Validation loss: {self.best_val_loss:.4f}")
                else:
                    self.early_stop_counter += 1
                    lprint(f"Validation loss not improved ({self.early_stop_counter}/{self.early_stop_patience})")
            else:
                val_loss = train_losses['total']
                is_best = val_loss < (self.best_val_loss - self.early_stop_min_delta)
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stop_counter = 0  
                else:
                    self.early_stop_counter += 1
                    lprint(f"Training loss not improved ({self.early_stop_counter}/{self.early_stop_patience})")
            
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
            
            for key, value in val_metrics.items():
                epoch_metrics[f'val_{key}'] = value
            
            metrics_history.append(epoch_metrics)
            
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
            
            if is_best:
                self.save_checkpoint(epoch, is_best)
                
                training_log['best_model_info'] = {
                    'epoch': epoch + 1,
                    'val_loss': self.best_val_loss,
                    'train_losses': train_losses,
                    'val_metrics': val_metrics,
                    'model_path': f"{self.save_path}_best.pth"
                }

            if self.early_stop_counter >= self.early_stop_patience:
                self.early_stopped = True
                lprint(f"Early stopping triggered! {self.early_stop_patience} epochs without significant improvement")
                lprint(f"Best validation loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch + 1})")
                break
        
        training_end_time = datetime.now()
        training_log['experiment_info']['end_time'] = training_end_time.isoformat()
        training_log['experiment_info']['total_training_time'] = str(training_end_time - training_start_time)
        training_log['experiment_info']['best_epoch'] = self.best_epoch + 1
        training_log['experiment_info']['best_val_loss'] = self.best_val_loss
        training_log['experiment_info']['early_stopped'] = self.early_stopped
        training_log['experiment_info']['early_stop_patience'] = self.early_stop_patience
        training_log['experiment_info']['early_stop_min_delta'] = self.early_stop_min_delta
        
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
            lprint(f"\nTraining early stopped! Best validation loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
            lprint(f"Early stopping reason: {self.early_stop_patience} epochs without significant improvement (threshold: {self.early_stop_min_delta})")
        else:
            lprint(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
        
        if performance_stats:
            lprint("\nDistillation training performance statistics:")
            if 'avg_train_batches_per_second' in performance_stats:
                lprint(f"Average training batches per second: {performance_stats['avg_train_batches_per_second']:.2f} "
                       f"(based on {performance_stats['total_train_batches']} batches)")
            if 'avg_val_batches_per_second' in performance_stats:
                lprint(f"Average validation batches per second: {performance_stats['avg_val_batches_per_second']:.2f} "
                       f"(based on {performance_stats['total_val_batches']} batches)")
        
        self.save_training_log(training_log)
        self.save_metrics_csv(metrics_history)
        
        if self.test_loader:
            checkpoint_path = f"{self.save_path}_best.pth"
            if os.path.exists(checkpoint_path):
                lprint(f"\nLoading best model for final test: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
                
                if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    lprint("Loaded learning rate scheduler state")
                
                if self.distill_loss.feature_adapter is not None and 'feature_adapter_state_dict' in checkpoint:
                    self.distill_loss.feature_adapter.load_state_dict(checkpoint['feature_adapter_state_dict'])
                    lprint("Loaded feature adapter")
                
                test_loss, test_metrics = self.test()
                
                test_performance = {}
                if self.all_test_batch_times:
                    avg_test_batch_time = np.mean(self.all_test_batch_times)
                    test_performance['avg_test_batch_time_seconds'] = avg_test_batch_time
                    test_performance['avg_test_batches_per_second'] = 1.0 / avg_test_batch_time
                    test_performance['total_test_batches'] = len(self.all_test_batch_times)
                    
                    lprint(f"Distillation test performance statistics:")
                    lprint(f"Average test batches per second: {test_performance['avg_test_batches_per_second']:.2f} "
                           f"(based on {test_performance['total_test_batches']} batches)")
                
                training_log['test_results'] = {
                    'test_loss': test_loss,
                    'test_metrics': test_metrics,
                    'test_time': datetime.now().isoformat(),
                    'test_performance': test_performance
                }
                
                self.save_training_log(training_log)
                
                return test_loss, test_metrics
            else:
                lprint(f"Warning: Best model file {checkpoint_path} not found, skipping test")
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    assert os.path.isfile(config_path), f'{config_path=}'
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")
    
    lprint(f"Config file loaded: {config_path}")
    lprint(json.dumps(config, indent=2))
    return config


def auto_load_distill_configs(teacher_type, student_type, dataset):
    if teacher_type == 'papagei' and student_type == 'mlp':
        distill_config_path = "config/distillation/papagei_to_mlp_distill.yaml"
    elif teacher_type == 'papagei' and student_type == 'linear':
        distill_config_path = "config/distillation/papagei_to_linear_distill.yaml"
    elif teacher_type == 'papagei' and student_type.startswith('gpt'):
        distill_config_path = "config/distillation/papagei_to_gpt_distill.yaml"
    elif teacher_type.startswith('gpt') and student_type.startswith('gpt'):
        distill_config_path = "config/distillation/gpt_to_gpt_patch_distill.yaml"
        if not os.path.exists(distill_config_path):
            distill_config_path = "config/distillation/gpt_to_mlp_distill.yaml"
            lprint(f"Warning: No GPT to GPT patch distillation config found, using default config")
    elif teacher_type.startswith('gpt') and student_type == 'mlp':
        distill_config_path = "config/distillation/gpt_to_mlp_distill.yaml"
    elif teacher_type.startswith('gpt') and student_type == 'linear':
        distill_config_path = "config/distillation/gpt_to_linear_distill.yaml"
    elif teacher_type == 'mlp' and student_type == 'linear':
        distill_config_path = "config/distillation/mlp_to_linear_distill.yaml"
    else:
        distill_config_path = "config/distillation/gpt_to_mlp_distill.yaml"
    
    if student_type in ['gpt_19m', 'gpt_1m']:
        student_config_path = f"config/models/gpt_config_{dataset}.yaml"
    else:
        student_config_path = f"config/models/{student_type}_config_{dataset}.yaml"
    
    data_config_path = f"config/data/{dataset}_data.yaml"
    
    distill_config = load_config(distill_config_path)
    student_model_config = load_config(student_config_path)
    data_config = load_config(data_config_path)
    
    config = {
        'distillation': distill_config['distillation'],
        'train_config': distill_config['train_config'],
        'data_config': data_config['data_config']
    }
    
    task_type = data_config['data_config'].get('task_type', 'regression')
    
    return config, student_model_config['model_config'], task_type


def auto_infer_teacher_path(teacher_type, dataset, save_dir):
    teacher_path = os.path.join(save_dir, f"{teacher_type}_{dataset}_best.pth")
    return teacher_path


def create_model(model_config, model_type='gpt_19m'):
    if model_type.lower() == 'papagei':
        try:
            model = create_papagei_model(model_config)
        except Exception as e:
            lprint(f"Failed to create Papagei model: {e}")
            model = PapageiModel(model_config)
    elif model_type.lower() in ['gpt', 'gpt_19m', 'gpt_1m']:
        if model_type.lower() == 'gpt_1m':
            gpt_1m_config_path = 'config/gpt_1M.json'
            if os.path.exists(gpt_1m_config_path):
                gpt_1m_config = load_config(gpt_1m_config_path)
                merged_config = model_config.copy()
                merged_config.update(gpt_1m_config)
            else:
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
            
            merged_config['PARAMS'] = '1M'
            model = GPT_with_linearOutput(merged_config)
        else:
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
        raise ValueError(f"Unsupported model type: {model_type}")
    
    lprint(f"Created {model_type.upper()} model:")
    lprint(model)
    return model


def load_teacher_model(teacher_path, teacher_type, device, n_patches=None):
    lprint(f"Loading teacher model: {teacher_path} (Type: {teacher_type})")
    
    checkpoint = torch.load(teacher_path, map_location='cpu')
    
    if 'dalia' in teacher_path:
        dataset = 'dalia'
    elif 'stanfordAF' in teacher_path:
        dataset = 'stanfordAF'
    else:
        dataset = 'dalia' 
    
    if teacher_type in ['gpt_19m', 'gpt_1m']:
        model_config_path = f"config/models/gpt_config_{dataset}.yaml"
    else:
        model_config_path = f"config/models/{teacher_type}_config_{dataset}.yaml"
    
    if os.path.exists(model_config_path):
        full_config = load_config(model_config_path)
        teacher_config = full_config['model_config'].copy()
        
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
    
        if n_patches is not None:
            teacher_config['n_patches'] = n_patches
        
        if teacher_type == 'gpt_1m':
            gpt_1m_config_path = 'config/gpt_1M.json'
            if os.path.exists(gpt_1m_config_path):
                gpt_1m_config = load_config(gpt_1m_config_path)
                for key in ['d_model', 'n_heads', 'n_layers', 'dropout', 'max_len']:
                    if key in gpt_1m_config:
                        teacher_config[key] = gpt_1m_config[key]
                teacher_config['PARAMS'] = '1M'
        elif teacher_type == 'gpt_19m':
            teacher_config['PARAMS'] = '19M'
    else:
        lprint(f"Warning: Config file {model_config_path} not found, using default config")
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
    
    teacher_model = create_model(teacher_config, teacher_type)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'student_model_state_dict' in checkpoint:
        state_dict = checkpoint['student_model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]  
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    teacher_model.load_state_dict(state_dict)
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  

    for param in teacher_model.parameters():
        param.requires_grad = False
    
    lprint("Teacher model loaded and frozen")
    return teacher_model


def count_parameters(model):
    if isinstance(model, PapageiModel):
        return count_papagei_parameters(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lprint(f"Total parameters: {total_params:,}")
    lprint(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


def main():
    parser = argparse.ArgumentParser(description='Unified PPG model knowledge distillation training')
    parser.add_argument('--teacher_type', type=str, required=True,
                      choices=['gpt_19m', 'gpt_1m', 'linear', 'mlp', 'papagei'], help='Type of teacher model')
    parser.add_argument('--student_type', type=str, required=True,
                      choices=['gpt_19m', 'gpt_1m', 'linear', 'mlp', 'papagei'], help='Type of student model')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['dalia', 'stanfordAF'], help='Dataset name')
    parser.add_argument('--teacher_path', type=str, help='Path to teacher model (optional, auto-inferred)')
    parser.add_argument('--save_dir', type=str, default='./output', help='Directory to save teacher model')
    parser.add_argument('--save_dir_student', type=str, default='./output_s', help='Directory to save student model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_only', action='store_true', help='Test only')
    parser.add_argument('--no-test', action='store_true', help='Disable automatic post-training test')
    
    parser.add_argument('--alpha', type=float, help='Prediction distillation loss weight (overrides config file)')
    parser.add_argument('--beta', type=float, help='Global feature distillation loss weight (overrides config file)')
    parser.add_argument('--gamma', type=float, help='Patch-level distillation loss weight (overrides config file)')
    parser.add_argument('--lr_max', type=float, help='Maximum learning rate (overrides config file)')
    parser.add_argument('--lr_init', type=str, help='Initial learning rate (overrides config file)')
    parser.add_argument('--lr_final', type=str, help='Final learning rate (overrides config file)')
    parser.add_argument('--lr_schedule_ratio', type=float, help='Learning rate scheduling ratio (overrides config file)')
    parser.add_argument('--lr_warm_up', type=float, help='Learning rate warmup ratio (overrides config file)')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (overrides config file)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config, student_config, task_type = auto_load_distill_configs(
        args.teacher_type, args.student_type, args.dataset
    )

    if args.alpha is not None:
        config['distillation']['alpha'] = args.alpha
        lprint(f"✅ Overrode alpha parameter: {args.alpha}")
    if args.beta is not None:
        config['distillation']['beta'] = args.beta
        lprint(f"✅ Overrode beta parameter: {args.beta}")
    if args.gamma is not None:
        config['distillation']['gamma'] = args.gamma
        lprint(f"✅ Overrode gamma parameter: {args.gamma}")
    if args.lr_max is not None:
        config['train_config']['lr_max'] = args.lr_max
        lprint(f"✅ Overrode lr_max parameter: {args.lr_max}")
    if args.lr_init is not None:
        config['train_config']['lr_init'] = args.lr_init
        lprint(f"✅ Overrode lr_init parameter: {args.lr_init}")
    if args.lr_final is not None:
        config['train_config']['lr_final'] = args.lr_final
        lprint(f"✅ Overrode lr_final parameter: {args.lr_final}")
    if args.lr_schedule_ratio is not None:
        config['train_config']['lr_schedule_ratio'] = args.lr_schedule_ratio
        lprint(f"✅ Overrode lr_schedule_ratio parameter: {args.lr_schedule_ratio}")
    if args.lr_warm_up is not None:
        config['train_config']['lr_warm_up'] = args.lr_warm_up
        lprint(f"✅ Overrode lr_warm_up parameter: {args.lr_warm_up}")
    if args.weight_decay is not None:
        config['train_config']['weight_decay'] = args.weight_decay
        lprint(f"✅ Overrode weight_decay parameter: {args.weight_decay}")
    
    if args.teacher_path is None:
        teacher_path = auto_infer_teacher_path(args.teacher_type, args.dataset, args.save_dir)
    else:
        teacher_path = args.teacher_path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprint(f"Using device: {device}")
    lprint(f"Dataset: {args.dataset}, Teacher: {args.teacher_type}, Student: {args.student_type}, Task type: {task_type}")
    
    os.makedirs(args.save_dir_student, exist_ok=True)
    
    save_path = os.path.join(args.save_dir_student, 
                           f"distill_{args.teacher_type}_to_{args.student_type}_{args.dataset}")
    
    data_cfg = config['data_config']
    lprint("Loading data...")

    patch_size = student_config.get('patch_size', 40)

    n_patches = calc_n_patches(data_cfg['train_data_path'], patch_size)
    lprint(f"Calculated number of patches: {n_patches}")
    
    student_config['n_patches'] = n_patches
    
    teacher_model = load_teacher_model(teacher_path, args.teacher_type, device, n_patches)
    lprint("Teacher model:")
    count_parameters(teacher_model)
    
    student_model = create_model(student_config, args.student_type)
    student_model = student_model.to(device)
    lprint("Student model:")
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
    
    if args.test_only:
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            student_model.load_state_dict(checkpoint['student_model_state_dict'])
            lprint(f"Loaded student model: {checkpoint_path}")
            
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
                    lprint("Loaded scheduler state")
                
                # 如果有特征适配器，加载它（全局和patch共用同一个）
                if trainer.distill_loss.feature_adapter is not None and 'feature_adapter_state_dict' in checkpoint:
                    trainer.distill_loss.feature_adapter.load_state_dict(checkpoint['feature_adapter_state_dict'])
                    lprint("Loaded feature adapter")
                
                test_loss, test_metrics = trainer.test()
                lprint(f"Test loss: {test_loss:.4f}")
                lprint(f"Test metrics: {test_metrics}")
            else:
                lprint("Test data not found")
        else:
            lprint(f"Model file not found: {checkpoint_path}")
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

    def update_training_log_info(trainer, teacher_type, student_type, dataset):
        original_train = trainer.train
        
        def enhanced_train():
            result = original_train()
            try:
                log_file = f"{trainer.save_path}_distill_training_log.json"
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        training_log = json.load(f)
                    
                    training_log['experiment_info']['teacher_type'] = teacher_type
                    training_log['experiment_info']['student_type'] = student_type
                    training_log['experiment_info']['dataset'] = dataset
                    training_log['experiment_info']['teacher_path'] = teacher_path
                    
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(training_log, f, indent=2, ensure_ascii=False)
            except Exception as e:
                lprint(f"Error updating log info: {e}")
            
            return result
        
        trainer.train = enhanced_train
    
    update_training_log_info(trainer, args.teacher_type, args.student_type, args.dataset)
    
    test_loss, test_metrics = trainer.train()
    
    if not args.no_test and test_loader and test_loss is not None:
        lprint(f"\nFinal test loss: {test_loss:.4f}")
        lprint(f"Final test metrics: {test_metrics}")
    elif not args.no_test and not test_loader:
        lprint("Warning:Auto-test enabled but no test data found")


if __name__ == "__main__":
    main() 