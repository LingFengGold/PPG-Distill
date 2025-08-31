#!/usr/bin/env python
"""
统一的PPG模型训练脚本
支持GPT、Linear、MLP模型的训练
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import csv

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn as nn
import torch.nn.functional as F

# 导入模型
from model.gpt import GPT_with_linearOutput
from model.linear import LinearModel, create_linear_model
from model.mlp import MLP, create_mlp_model
from model.papagei import PapageiModel, create_papagei_model, count_papagei_parameters
from data.pretrain_dataset import PretrainDataset
try:
    from trainer.loaders import load_train_objs, load_trainer
except ImportError:
    pass  # 如果导入失败，使用内建功能
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


def auto_load_configs(model_type, dataset):
    """根据模型类型和数据集自动加载配置文件"""
    # 推断模型配置文件路径
    if model_type in ['gpt_19m', 'gpt_1m']:
        model_config_path = f"config/models/gpt_config_{dataset}.yaml"
    else:
        model_config_path = f"config/models/{model_type}_config_{dataset}.yaml"
    
    # 数据配置文件路径
    data_config_path = f"config/data/{dataset}_data.yaml"
    
    # 加载配置
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    
    # 合并配置
    config = {
        'model_config': model_config['model_config'],
        'train_config': model_config['train_config'],
        'data_config': data_config['data_config']
    }
    
    # 获取任务类型
    task_type = data_config['data_config'].get('task_type', 'regression')
    
    return config, task_type


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
                if key not in model_config:
                    model_config[key] = default_value
                    
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


def setup_optimizer(model, train_config):
    """设置优化器"""
    optimizer_type = train_config.get('optimizer', 'Adam').lower()
    lr = train_config['lr_max']
    weight_decay = train_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    lprint(f"优化器: {optimizer_type}, 学习率: {lr}, 权重衰减: {weight_decay}")
    return optimizer


def setup_scheduler(optimizer, train_config):
    """设置学习率调度器"""
    scheduler_type = train_config.get('scheduler', None)
    
    if scheduler_type is None:
        return None
    elif scheduler_type.lower() == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        T_max = train_config['epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type.lower() == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = train_config.get('step_size', 10)
        gamma = train_config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        lprint(f"警告: 未识别的调度器类型 {scheduler_type}, 使用默认调度器")
        scheduler = None
    
    return scheduler


def calculate_metrics(predictions, targets, task_type='regression'):
    """计算评估指标"""
    # 确保输入是numpy数组
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if task_type == 'regression':
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        return {'mse': mse, 'mae': mae}
    else:  # classification
        from sklearn.metrics import accuracy_score, f1_score
        pred_classes = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        acc = accuracy_score(targets, pred_classes)
        f1 = f1_score(targets, pred_classes, average='binary')
        return {'accuracy': acc, 'f1': f1}


def train_epoch(model, train_loader, optimizer, criterion, device, task_type='regression', collect_batch_times=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    # 时间统计
    batch_times = []
    epoch_start_time = time.time()
    
    progress_bar = tqdm.tqdm(train_loader, desc='Training')
    
    for batch_idx, data in enumerate(progress_bar):
        batch_start_time = time.time()
        
        ppg_segments = data["ppg_segments"].to(device)
        
        if task_type == 'regression':
            targets = data["ft_label"].to(device).float()
        else:  # classification
            targets = data["ft_label"].to(device).long()

        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(ppg_segments)
        
        if task_type == 'regression':
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)
        else:  # classification
            loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # 收集预测和目标用于计算指标
        if task_type == 'regression':
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        else:
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        # 记录batch时间
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # 收集批次时间用于全局统计
        if collect_batch_times is not None:
            collect_batch_times.append(batch_time)
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Batch/s': f'{1.0/batch_time:.2f}'
        })
    
    avg_loss = total_loss / num_batches
    metrics = calculate_metrics(all_predictions, all_targets, task_type)
    
    # 计算时间统计
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times)
    batches_per_second = 1.0 / avg_batch_time
    
    lprint(f"训练统计 - 总时间: {epoch_time:.2f}s, 平均每批: {avg_batch_time:.5f}s, 批次/秒: {batches_per_second:.2f}")
    
    return avg_loss, metrics


def validate_epoch(model, val_loader, criterion, device, task_type='regression', collect_batch_times=None):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    # 时间统计
    batch_times = []
    epoch_start_time = time.time()
    
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, desc='Validation'):
            batch_start_time = time.time()
            
            ppg_segments = data["ppg_segments"].to(device)
            
            if task_type == 'regression':
                targets = data["ft_label"].to(device).float()
            else:  # classification
                targets = data["ft_label"].to(device).long()
            
            # 前向传播
            outputs = model(ppg_segments)
            
            if task_type == 'regression':
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
            else:  # classification
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测和目标
            if task_type == 'regression':
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else:
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            # 记录batch时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 收集批次时间用于全局统计
            if collect_batch_times is not None:
                collect_batch_times.append(batch_time)
    
    avg_loss = total_loss / num_batches
    metrics = calculate_metrics(all_predictions, all_targets, task_type)
    
    # 计算时间统计
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times)
    batches_per_second = 1.0 / avg_batch_time
    
    lprint(f"验证统计 - 总时间: {epoch_time:.2f}s, 平均每批: {avg_batch_time:.5f}s, 批次/秒: {batches_per_second:.2f}")
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, best_loss, config, save_path, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'config': config
    }
    
    # 只保存最佳模型
    if is_best:
        torch.save(checkpoint, f"{save_path}_best.pth")
        lprint(f"保存最佳模型: {save_path}_best.pth")


def save_training_log(log_data, save_path):
    """保存训练日志到JSON文件"""
    log_file = f"{save_path}_training_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    lprint(f"训练日志已保存: {log_file}")


def save_metrics_csv(metrics_history, save_path):
    """保存训练指标到CSV文件"""
    csv_file = f"{save_path}_metrics.csv"
    
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
    
    lprint(f"训练指标CSV已保存: {csv_file}")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='统一PPG模型训练')
    parser.add_argument('--model_type', type=str, required=True, 
                      choices=['gpt_19m', 'gpt_1m', 'linear', 'mlp', 'papagei'], help='模型类型')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['dalia', 'stanfordAF'], help='数据集名称')
    parser.add_argument('--save_dir', type=str, default='./output', help='保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--eval_only', action='store_true', help='只进行评估')
    parser.add_argument('--test_only', action='store_true', help='只进行测试')
    parser.add_argument('--no-test', action='store_true', help='禁用训练完成后的自动测试')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 自动加载配置
    config, task_type = auto_load_configs(args.model_type, args.dataset)
    model_config = config['model_config']
    train_config = config['train_config']
    data_config = config['data_config']
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprint(f"使用设备: {device}")
    lprint(f"数据集: {args.dataset}, 模型: {args.model_type}, 任务类型: {task_type}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_type}_{args.dataset}")
    
    # 数据加载和n_patches计算
    lprint("加载数据...")
    # 获取patch_size
    patch_size = model_config.get('patch_size', 40)
    
    n_patches = calc_n_patches(data_config['train_data_path'], patch_size)
    lprint(f"计算得到的patch数量: {n_patches}")
    model_config['n_patches'] = n_patches
    
    # 创建模型
    model = create_model(model_config, args.model_type)
    model = model.to(device)
    count_parameters(model)
    
    train_dataset = PretrainDataset(
        data_config['train_data_path'],
        patch_size=patch_size,
        train_labels_dataset_path=data_config.get('train_label_path', ''),
        data_red_factor=data_config.get('data_red_factor', 1)
    )
    
    if data_config.get('val_data_path'):
        val_dataset = PretrainDataset(
            data_config['val_data_path'],
            patch_size=patch_size,
            train_labels_dataset_path=data_config.get('val_label_path', ''),
            data_red_factor=data_config.get('data_red_factor', 1)
        )
    else:
        val_dataset = None
    
    # 测试数据集
    if data_config.get('test_data_path'):
        test_dataset = PretrainDataset(
            data_config['test_data_path'],
            patch_size=patch_size,
            train_labels_dataset_path=data_config.get('test_label_path', ''),
            data_red_factor=data_config.get('data_red_factor', 1)
        )
    else:
        test_dataset = None
    
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
    
    # 设置损失函数
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:  # classification
        criterion = nn.CrossEntropyLoss()
    
    # 如果只是评估，加载模型并评估
    if args.eval_only:
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 处理不同格式的checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'student_model_state_dict' in checkpoint:
                state_dict = checkpoint['student_model_state_dict']
            elif 'model' in checkpoint:
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
            
            model.load_state_dict(state_dict)
            lprint(f"已加载模型: {checkpoint_path}")
            
            if val_loader:
                eval_batch_times = []
                val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, task_type, eval_batch_times)
                lprint(f"验证损失: {val_loss:.4f}")
                lprint(f"验证指标: {val_metrics}")
                
                # 显示评估性能统计
                if eval_batch_times:
                    avg_eval_batch_time = np.mean(eval_batch_times)
                    avg_eval_batches_per_second = 1.0 / avg_eval_batch_time
                    lprint(f"评估性能统计:")
                    lprint(f"平均评估批次/秒: {avg_eval_batches_per_second:.2f} "
                           f"(基于 {len(eval_batch_times)} 个批次)")
        else:
            lprint(f"未找到模型文件: {checkpoint_path}")
        return
    
    # 如果只是测试，加载模型并测试
    if args.test_only:
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 处理不同格式的checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'student_model_state_dict' in checkpoint:
                state_dict = checkpoint['student_model_state_dict']
            elif 'model' in checkpoint:
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
            
            model.load_state_dict(state_dict)
            lprint(f"已加载模型: {checkpoint_path}")
            
            if test_loader:
                test_only_batch_times = []
                test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, task_type, test_only_batch_times)
                lprint(f"测试损失: {test_loss:.4f}")
                lprint(f"测试指标: {test_metrics}")
                
                # 显示测试性能统计
                if test_only_batch_times:
                    avg_test_batch_time = np.mean(test_only_batch_times)
                    avg_test_batches_per_second = 1.0 / avg_test_batch_time
                    lprint(f"测试性能统计:")
                    lprint(f"平均测试批次/秒: {avg_test_batches_per_second:.2f} "
                           f"(基于 {len(test_only_batch_times)} 个批次)")
            else:
                lprint("未找到测试数据")
        else:
            lprint(f"未找到模型文件: {checkpoint_path}")
        return
    
    # 设置优化器和调度器
    optimizer = setup_optimizer(model, train_config)
    scheduler = setup_scheduler(optimizer, train_config)
    
    # 训练循环
    epochs = train_config['epochs']
    best_val_loss = float('inf')
    best_epoch = 0
    
    # 初始化日志记录
    training_start_time = datetime.now()
    metrics_history = []
    
    # 初始化性能统计
    all_train_batch_times = []
    all_val_batch_times = []
    all_test_batch_times = []
    
    training_log = {
        'experiment_info': {
            'model_type': args.model_type,
            'dataset': args.dataset,
            'task_type': task_type,
            'start_time': training_start_time.isoformat(),
            'config': config,
            'total_params': count_parameters(model)[0],
            'trainable_params': count_parameters(model)[1]
        },
        'training_history': [],
        'best_model_info': {},
        'performance_stats': {}
    }
    
    lprint("开始训练...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        lprint(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, task_type, all_train_batch_times
        )
        
        lprint(f"训练损失: {train_loss:.4f}")
        lprint(f"训练指标: {train_metrics}")
        
        # 验证
        val_loss = None
        val_metrics = {}
        if val_loader:
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, task_type, all_val_batch_times
            )
            lprint(f"验证损失: {val_loss:.4f}")
            lprint(f"验证指标: {val_metrics}")
            
            # 检查是否为最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                lprint(f"新的最佳模型! 验证损失: {best_val_loss:.4f}")
        else:
            val_loss = train_loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
        
        # 记录当前epoch的指标
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0] if scheduler else train_config['lr_max']
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
            'is_best': is_best
        }
        
        # 添加训练指标
        for key, value in train_metrics.items():
            epoch_metrics[f'train_{key}'] = value
        
        # 添加验证指标
        for key, value in val_metrics.items():
            epoch_metrics[f'val_{key}'] = value
        
        metrics_history.append(epoch_metrics)
        
        # 记录到训练日志
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
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
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, config, save_path, is_best
            )
            
            # 更新最佳模型信息
            training_log['best_model_info'] = {
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_path': f"{save_path}_best.pth"
            }
        
        # 更新学习率
        if scheduler:
            scheduler.step()
            lprint(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
    
    # 训练结束，更新日志
    training_end_time = datetime.now()
    training_log['experiment_info']['end_time'] = training_end_time.isoformat()
    training_log['experiment_info']['total_training_time'] = str(training_end_time - training_start_time)
    training_log['experiment_info']['best_epoch'] = best_epoch + 1
    training_log['experiment_info']['best_val_loss'] = best_val_loss
    
    # 计算性能统计
    performance_stats = {}
    if all_train_batch_times:
        avg_train_batch_time = np.mean(all_train_batch_times)
        performance_stats['avg_train_batch_time_seconds'] = avg_train_batch_time
        performance_stats['avg_train_batches_per_second'] = 1.0 / avg_train_batch_time
        performance_stats['total_train_batches'] = len(all_train_batch_times)
    
    if all_val_batch_times:
        avg_val_batch_time = np.mean(all_val_batch_times)
        performance_stats['avg_val_batch_time_seconds'] = avg_val_batch_time
        performance_stats['avg_val_batches_per_second'] = 1.0 / avg_val_batch_time
        performance_stats['total_val_batches'] = len(all_val_batch_times)
    
    training_log['performance_stats'] = performance_stats
    
    lprint(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch+1})")
    
    # 显示性能统计
    if performance_stats:
        lprint("\n性能统计:")
        if 'avg_train_batches_per_second' in performance_stats:
            lprint(f"平均训练批次/秒: {performance_stats['avg_train_batches_per_second']:.2f} "
                   f"(基于 {performance_stats['total_train_batches']} 个批次)")
        if 'avg_val_batches_per_second' in performance_stats:
            lprint(f"平均验证批次/秒: {performance_stats['avg_val_batches_per_second']:.2f} "
                   f"(基于 {performance_stats['total_val_batches']} 个批次)")
    
    # 保存训练日志和指标CSV
    save_training_log(training_log, save_path)
    save_metrics_csv(metrics_history, save_path)
    
    # 训练完成后进行测试（除非禁用）
    if not args.no_test and test_loader:
        lprint("\n开始最终测试...")
        # 加载最佳模型
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            lprint(f"已加载最佳模型: {checkpoint_path}")
            
            # 进行测试
            test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, task_type, all_test_batch_times)
            lprint(f"最终测试损失: {test_loss:.4f}")
            lprint(f"最终测试指标: {test_metrics}")
            
            # 计算测试性能统计
            test_performance = {}
            if all_test_batch_times:
                avg_test_batch_time = np.mean(all_test_batch_times)
                test_performance['avg_test_batch_time_seconds'] = avg_test_batch_time
                test_performance['avg_test_batches_per_second'] = 1.0 / avg_test_batch_time
                test_performance['total_test_batches'] = len(all_test_batch_times)
                
                lprint(f"测试性能统计:")
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
            save_training_log(training_log, save_path)
            
        else:
            lprint(f"未找到最佳模型文件: {checkpoint_path}")
            training_log['test_results'] = {
                'error': 'Best model file not found',
                'test_time': datetime.now().isoformat()
            }
            save_training_log(training_log, save_path)
    elif not args.no_test and not test_loader:
        lprint("警告: 启用了自动测试但未找到测试数据")
        training_log['test_results'] = {
            'warning': 'Test enabled but no test data found',
            'test_time': datetime.now().isoformat()
        }
        save_training_log(training_log, save_path)


if __name__ == "__main__":
    main() 