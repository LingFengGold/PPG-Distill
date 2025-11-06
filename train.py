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
    pass  
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


def auto_load_configs(model_type, dataset):
    if model_type in ['gpt_19m', 'gpt_1m']:
        model_config_path = f"config/models/gpt_config_{dataset}.yaml"
    else:
        model_config_path = f"config/models/{model_type}_config_{dataset}.yaml"
    
    data_config_path = f"config/data/{dataset}_data.yaml"
    
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    
    config = {
        'model_config': model_config['model_config'],
        'train_config': model_config['train_config'],
        'data_config': data_config['data_config']
    }
    
    task_type = data_config['data_config'].get('task_type', 'regression')
    
    return config, task_type


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


def count_parameters(model):
    if isinstance(model, PapageiModel):
        return count_papagei_parameters(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lprint(f"Total parameters: {total_params:,}")
    lprint(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


def setup_optimizer(model, train_config):
    optimizer_type = train_config.get('optimizer', 'Adam').lower()
    lr = train_config['lr_max']
    weight_decay = train_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    lprint(f"Optimizer: {optimizer_type}, learning rate: {lr}, weight decay: {weight_decay}")
    return optimizer


def setup_scheduler(optimizer, train_config):
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
        lprint(f"Warning: Unrecognized scheduler type {scheduler_type}, using default scheduler")
        scheduler = None
    
    return scheduler


def calculate_metrics(predictions, targets, task_type='regression'):
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
    model.train()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
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
        
        outputs = model(ppg_segments)
        
        if task_type == 'regression':
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)
        else:  # classification
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if task_type == 'regression':
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        else:
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        if collect_batch_times is not None:
            collect_batch_times.append(batch_time)
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Batch/s': f'{1.0/batch_time:.2f}'
        })
    
    avg_loss = total_loss / num_batches
    metrics = calculate_metrics(all_predictions, all_targets, task_type)
    
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times)
    batches_per_second = 1.0 / avg_batch_time
    
    lprint(f"Training statistics - Total time: {epoch_time:.2f}s, Average batch time: {avg_batch_time:.5f}s, Batches/second: {batches_per_second:.2f}")
    
    return avg_loss, metrics


def validate_epoch(model, val_loader, criterion, device, task_type='regression', collect_batch_times=None):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
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
            
            outputs = model(ppg_segments)
            
            if task_type == 'regression':
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
            else:  # classification
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            if task_type == 'regression':
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else:
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if collect_batch_times is not None:
                collect_batch_times.append(batch_time)
    
    avg_loss = total_loss / num_batches
    metrics = calculate_metrics(all_predictions, all_targets, task_type)
    
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times)
    batches_per_second = 1.0 / avg_batch_time
    
    lprint(f"Validation statistics - Total time: {epoch_time:.2f}s, Average batch time: {avg_batch_time:.5f}s, Batches/second: {batches_per_second:.2f}")
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, best_loss, config, save_path, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'config': config
    }
    
    if is_best:
        torch.save(checkpoint, f"{save_path}_best.pth")
        lprint(f"Saved best model: {save_path}_best.pth")


def save_training_log(log_data, save_path):
    log_file = f"{save_path}_training_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    lprint(f"Training log saved: {log_file}")


def save_metrics_csv(metrics_history, save_path):
    csv_file = f"{save_path}_metrics.csv"
    
    if not metrics_history:
        return
    
    all_keys = set()
    for epoch_data in metrics_history:
        all_keys.update(epoch_data.keys())
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(metrics_history)
    
    lprint(f"Training metrics CSV saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='PPG model training')
    parser.add_argument('--model_type', type=str, required=True, 
                      choices=['gpt_19m', 'gpt_1m', 'linear', 'mlp', 'papagei'], help='Model type')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['dalia', 'stanfordAF'], help='Dataset name')
    parser.add_argument('--save_dir', type=str, default='./output', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
    parser.add_argument('--test_only', action='store_true', help='Only test')
    parser.add_argument('--no-test', action='store_true', help='Disable automatic test after training')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config, task_type = auto_load_configs(args.model_type, args.dataset)
    model_config = config['model_config']
    train_config = config['train_config']
    data_config = config['data_config']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprint(f"Using device: {device}")
    lprint(f"Dataset: {args.dataset}, Model: {args.model_type}, Task type: {task_type}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_type}_{args.dataset}")
    
    lprint("Loading data...")
    patch_size = model_config.get('patch_size', 40)
    
    n_patches = calc_n_patches(data_config['train_data_path'], patch_size)
    lprint(f"Calculated number of patches: {n_patches}")
    model_config['n_patches'] = n_patches
    
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

    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:  # classification
        criterion = nn.CrossEntropyLoss()
    
    if args.eval_only:
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
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
            
            model.load_state_dict(state_dict)
            lprint(f"Model loaded: {checkpoint_path}")
            
            if val_loader:
                eval_batch_times = []
                val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, task_type, eval_batch_times)
                lprint(f"Validation loss: {val_loss:.4f}")
                lprint(f"Validation metrics: {val_metrics}")
                
                if eval_batch_times:
                    avg_eval_batch_time = np.mean(eval_batch_times)
                    avg_eval_batches_per_second = 1.0 / avg_eval_batch_time
                    lprint(f"Evaluation performance statistics:")
                    lprint(f"Average evaluation batches per second: {avg_eval_batches_per_second:.2f} "
                           f"(based on {len(eval_batch_times)} batches)")
        else:
            lprint(f"Model file not found: {checkpoint_path}")
        return
    
    if args.test_only:
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
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
            
            model.load_state_dict(state_dict)
            lprint(f"Model loaded: {checkpoint_path}")
            
            if test_loader:
                test_only_batch_times = []
                test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, task_type, test_only_batch_times)
                lprint(f"Test loss: {test_loss:.4f}")
                lprint(f"Test metrics: {test_metrics}")
                
                if test_only_batch_times:
                    avg_test_batch_time = np.mean(test_only_batch_times)
                    avg_test_batches_per_second = 1.0 / avg_test_batch_time
                    lprint(f"Test performance statistics:")
                    lprint(f"Average test batches per second: {avg_test_batches_per_second:.2f} "
                           f"(based on {len(test_only_batch_times)} batches)")
            else:
                lprint("Test data not found")
        else:
            lprint(f"Model file not found: {checkpoint_path}")
        return
    
    optimizer = setup_optimizer(model, train_config)
    scheduler = setup_scheduler(optimizer, train_config)
    
    epochs = train_config['epochs']
    best_val_loss = float('inf')
    best_epoch = 0
    
    training_start_time = datetime.now()
    metrics_history = []
    
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
    
    lprint("Training started...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        lprint(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, task_type, all_train_batch_times
        )
        
        lprint(f"Training loss: {train_loss:.4f}")
        lprint(f"Training metrics: {train_metrics}")
        
        val_loss = None
        val_metrics = {}
        if val_loader:
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, task_type, all_val_batch_times
            )
            lprint(f"Validation loss: {val_loss:.4f}")
            lprint(f"Validation metrics: {val_metrics}")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                lprint(f"New best model! Validation loss: {best_val_loss:.4f}")
        else:
            val_loss = train_loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
        
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
        
        for key, value in train_metrics.items():
            epoch_metrics[f'train_{key}'] = value
        
        for key, value in val_metrics.items():
            epoch_metrics[f'val_{key}'] = value
        
        metrics_history.append(epoch_metrics)
        
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
        
        if is_best:
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, config, save_path, is_best
            )
            
            training_log['best_model_info'] = {
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_path': f"{save_path}_best.pth"
            }

        if scheduler:
            scheduler.step()
            lprint(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    training_end_time = datetime.now()
    training_log['experiment_info']['end_time'] = training_end_time.isoformat()
    training_log['experiment_info']['total_training_time'] = str(training_end_time - training_start_time)
    training_log['experiment_info']['best_epoch'] = best_epoch + 1
    training_log['experiment_info']['best_val_loss'] = best_val_loss
    
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
    
    lprint(f"\nTraining completed! Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch+1})")
    
    # 显示性能统计
    if performance_stats:
        lprint("\nPerformance statistics:")
        if 'avg_train_batches_per_second' in performance_stats:
            lprint(f"Average training batches per second: {performance_stats['avg_train_batches_per_second']:.2f} "
                   f"(based on {performance_stats['total_train_batches']} batches)")
        if 'avg_val_batches_per_second' in performance_stats:
            lprint(f"Average validation batches per second: {performance_stats['avg_val_batches_per_second']:.2f} "
                   f"(based on {performance_stats['total_val_batches']} batches)")
    
    save_training_log(training_log, save_path)
    save_metrics_csv(metrics_history, save_path)

    if not args.no_test and test_loader:
        lprint("\nFinal test started...")
        checkpoint_path = f"{save_path}_best.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            lprint(f"Best model loaded: {checkpoint_path}")
            
            test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, task_type, all_test_batch_times)
            lprint(f"Final test loss: {test_loss:.4f}")
            lprint(f"Final test metrics: {test_metrics}")
            
            test_performance = {}
            if all_test_batch_times:
                avg_test_batch_time = np.mean(all_test_batch_times)
                test_performance['avg_test_batch_time_seconds'] = avg_test_batch_time
                test_performance['avg_test_batches_per_second'] = 1.0 / avg_test_batch_time
                test_performance['total_test_batches'] = len(all_test_batch_times)
                
                lprint(f"Test performance statistics:")
                lprint(f"Average test batches per second: {test_performance['avg_test_batches_per_second']:.2f} "
                       f"(based on {test_performance['total_test_batches']} batches)")
            
            training_log['test_results'] = {
                'test_loss': test_loss,
                'test_metrics': test_metrics,
                'test_time': datetime.now().isoformat(),
                'test_performance': test_performance
            }
            
            save_training_log(training_log, save_path)
            
        else:
            lprint(f"Best model file not found: {checkpoint_path}")
            training_log['test_results'] = {
                'error': 'Best model file not found',
                'test_time': datetime.now().isoformat()
            }
            save_training_log(training_log, save_path)
    elif not args.no_test and not test_loader:
        lprint("Warning: Automatic test enabled but no test data found")
        training_log['test_results'] = {
            'warning': 'Test enabled but no test data found',
            'test_time': datetime.now().isoformat()
        }
        save_training_log(training_log, save_path)


if __name__ == "__main__":
    main() 