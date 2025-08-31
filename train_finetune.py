#!/usr/bin/env python

import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import sys
sys.path.append('.')
import argparse
import torch
assert torch.cuda.is_available()
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = False
import logging
logging.basicConfig(level=logging.INFO)
from logging import info as lprint
import yaml
import json
import time
from pathlib import Path
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc, matthews_corrcoef, roc_auc_score, average_precision_score

from torch.utils.data import DistributedSampler, WeightedRandomSampler

from local.supp_fxns import *
from model.gpt import GPT_with_linearOutput
from trainer.loaders import *


def load_json(filepath):
    assert os.path.isfile(filepath), f'{filepath=}'
    assert filepath.endswith('.json'), f'{filepath=}'
    with open(filepath, 'r') as f:
        config = json.load(f)
    lprint(json.dumps(config, indent=1))     # better printing
    return config


def load_yaml(filepath):
    assert os.path.isfile(filepath), f'{filepath=}'
    assert filepath.endswith('.yaml'), f'{filepath=}'
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    lprint(json.dumps(config, indent=1))
    return config


def calc_n_patches(filepath, fs):
    assert os.path.isfile(filepath), f'{filepath=}'
    seq_length = get_numpy_array_metadata(filepath, return_attrs=True)[0][-1]
    assert seq_length >= fs, 'need atleast 1 second signal length'
    assert seq_length % fs == 0, 'signal length (in seconds) must be a whole number'
    n_patches = seq_length // fs
    return n_patches


def get_latest_file(directory: str, pattern: str) -> Optional[Path]:
    """
    Retrieves the latest file in the specified directory matching the given glob pattern.

    Parameters:
        directory (str): The path to the directory to search in.
        pattern (str): The glob pattern to match files (e.g., "*.txt", "data_*.csv").

    Returns:
        Path or None: The Path object of the latest file if found; otherwise, None.
    """
    # Initialize the Path object for the directory
    dir_path = Path(directory)

    # Validate if the directory exists
    if not dir_path.is_dir():
        raise ValueError(f"The directory '{directory}' does not exist or is not a directory.")

    # Use glob to find all files matching the pattern
    matched_files = list(dir_path.glob(pattern))

    # If no files match the pattern, return None
    if not matched_files:
        return None

    # Use max with key as the modification time to find the latest file
    latest_file = max(matched_files, key=lambda f: f.stat().st_mtime)

    return str(latest_file)


def load_state_dict_gpt_custom(model, ckpt_path):
    assert os.path.exists(ckpt_path), f'FAILED TO LOAD (path looks correct?): {ckpt_path=}'
    model = model.to(torch.device('cpu'))
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    for key in list(state_dict['model']):
        if key.startswith('_orig_mod.'):
            state_dict['model']['.'.join(key.split('.')[1:])] = state_dict['model'].pop(key)
    model.load_state_dict(state_dict['model'])
    lprint(f'state dict from {ckpt_path} loaded onto model')


def do_train():
    # define some variables
    fs = 40
    device = torch.device('cuda')

    # load config files
    finetune_config = load_yaml(finetune_config_path)
    train_config_path = f'./config/training_config_gpt_ft_{finetune_config["tune_mode"]}.json'
#    train_config_path = f'./config/training_config_gpt_ft_head.json'
    train_config = load_json(train_config_path)

    #
    PARAMS = finetune_config['model_size']
    task_type = finetune_config['task']
    assert task_type in ['classification', 'regression']
    model_config_path = f'./config/gpt_{PARAMS}.json'
    model_config = load_json(model_config_path)

    ## modify model_config
    #
    if task_type == 'classification':
        model_config['loss'] = 'ce+laplace'
    elif task_type == 'regression':
        model_config['loss'] = 'mse+laplace'
    else:
        raise NotImplementedError(f'{task_type=}')
    #
    model_config['PARAMS'] = PARAMS
    #
    model_config['n_patches'] = calc_n_patches(finetune_config['train_data_path'], fs)
    #
    assert finetune_config['tune_mode'] in ['full', 'head'], f"{finetune_config['tune_mode']=}"
    model_config['tune_mode'] = finetune_config['tune_mode']
    # TODO - ECG metrics
    try:
        model_config['multi_target_regression'] = finetune_config['multi_target_regression']
    except Exception as e:
        lprint(e)
        model_config['multi_target_regression'] = False
    if model_config['multi_target_regression'] is True: # TODO
        assert task_type == 'regression'
        train_label_path = convert_to_path_list(finetune_config['train_label_path'], validate=True)
        test_label_path = convert_to_path_list(finetune_config['test_label_path'], validate=True)
        assert len(train_label_path) == len(test_label_path), f'{len(train_label_path)=} {len(train_label_path)=}'
        n_regression_tasks = len(train_label_path)
        assert n_regression_tasks > 1
    else:
        train_label_path = finetune_config['train_label_path']
        test_label_path = finetune_config['test_label_path']
        if task_type == 'regression':
            n_regression_tasks = 1
    #
    if task_type == 'classification':
        model_config['out_classes'] = np.unique(np.load(finetune_config['train_label_path'])).shape[0]
        assert model_config['out_classes'] <= 35, f"{model_config['out_classes']} are you sure we have classes more than 10?"
    elif task_type == 'regression':
        model_config['out_classes'] = n_regression_tasks
    else:
        raise NotImplementedError(f'{task_type=}')
    try:
        model_config['apply_mask'] = train_config['apply_mask']
    except Exception as e:
        lprint(e)
        model_config['apply_mask'] = False
    try:
        model_config['use_penultimate_layer'] = finetune_config['use_penultimate_layer']
    except Exception as e:
        lprint(e)
        model_config['use_penultimate_layer'] = False
    try:
        model_config['is_input_multichannel'] = finetune_config['is_input_multichannel']
    except Exception as e:
        lprint(e)
        model_config['is_input_multichannel'] = False
#    if model_config['is_input_multichannel'] == True:
#        n_channel_data = get_numpy_array_metadata(finetune_config['train_data_path'], return_attrs=True)[0]
#        assert len(n_channel_data) == 3, f'{n_channel_data=} has channel dimension missing maybe'
#        n_channel_data = n_channel_data[1]
##        n_channel_data_val = get_numpy_array_metadata(finetune_config['val_data_path'], return_attrs=True)[0]  # skipping val data checking since we do not utilize val data yet
##        assert len(n_channel_data_val) == 3
##        assert n_channel_data_val[1] == n_channel_data
#        n_channel_data_test = get_numpy_array_metadata(finetune_config['test_data_path'], return_attrs=True)[0]
#        assert len(n_channel_data_test) == 3
#        assert n_channel_data_test[1] == n_channel_data
#    else:
#        n_channel_data = get_numpy_array_metadata(finetune_config['train_data_path'], return_attrs=True)[0]
#        assert len(n_channel_data) == 2
#        n_channel_data = 2  # by default, we work with single channel data
##        n_channel_data_val = get_numpy_array_metadata(finetune_config['val_data_path'], return_attrs=True)[0]
##        assert len(n_channel_data_val) == 2
#        n_channel_data_test = get_numpy_array_metadata(finetune_config['test_data_path'], return_attrs=True)[0]
#        assert len(n_channel_data_test) == 2
#    model_config['n_channel_data'] = n_channel_data
    train_data_path = finetune_config['train_data_path']
    #
    try:
        model_config['strict_loading_gpt_state_dict'] = finetune_config['strict_loading_gpt_state_dict']
    except Exception as e:
        lprint(e)
        model_config['strict_loading_gpt_state_dict'] = True
    # TODO - add ECG channel
    try:
        model_config['ecg_input'] = finetune_config['ecg_input']
    except Exception as e:
        lprint(e)
        model_config['ecg_input'] = False
    try:
        train_data_path_ecg = finetune_config['train_data_path_ecg']
    except Exception as e:
        lprint(e)
        train_data_path_ecg = None
    if train_data_path_ecg:
        assert os.path.exists(train_data_path_ecg)
    try:
        test_data_path_ecg = finetune_config['test_data_path_ecg']
    except Exception as e:
        lprint(e)
        test_data_path_ecg = None
    if test_data_path_ecg:
        assert os.path.exists(test_data_path_ecg)
    if model_config['ecg_input']:
        train_config['ecg_input'] = True
        assert train_data_path_ecg
        assert test_data_path_ecg
        ecg_data_len = get_numpy_array_metadata(train_data_path_ecg, return_attrs=True)[0][-1]
        ppg_data_len = get_numpy_array_metadata(train_data_path, return_attrs=True)[0][-1]
        ecg_patch_size = 40
        assert (ecg_data_len / ppg_data_len) == (ecg_patch_size / 40), f'{ecg_data_len=} {ppg_data_len=} unsupported ratio={ecg_data_len/ppg_data_len}'
    #
    try:
        model_config['model_name'] = finetune_config['model_name']
    except Exception as e:
        lprint(e)
        model_config['model_name'] = 'gpt'
    assert model_config['model_name'] in ['gpt', 'sqigpt', 'accxgpt', 'accygpt', 'acczgpt', 'accgpt', 'sqiaccgpt']
    try:
        model_config['gpt_state_dict_path'] = finetune_config['gpt_state_dict_path']
    except Exception as e:
        lprint(e)
        model_config['gpt_state_dict_path'] = None
    if model_config['gpt_state_dict_path'] is None:
        assert model_config['model_name'] == 'gpt', f"{model_config['model_name']=} {model_config['gpt_state_dict_path']=}"
    # fuse tuning related stuff
    try:
        model_config['fuse_tuning'] = finetune_config['fuse_tuning']
    except Exception as e:
        lprint(e)
        model_config['fuse_tuning'] = False
    try:
        model_config['fuse_feat_type'] = finetune_config['fuse_feat_type']
    except Exception as e:
        lprint(e)
        model_config['fuse_feat_type'] = None
    assert not (model_config['fuse_tuning'] and not model_config['fuse_feat_type']), f"{model_config['fuse_tuning']=} {model_config['fuse_feat_type']=} (fuse tuning is ON and fuse feats not provided)"
#    assert not (not model_config['fuse_tuning'] and model_config['fuse_feat_type']), f"{model_config['fuse_tuning']=} {model_config['fuse_feat_type']=} (dont provide fuse feats if not doing fuse tuning)"
    ## lora
    assign_key_to_dict_from_dict_safely(model_config, 'use_lora', finetune_config, False, lprint)    # this is just new way of doing what i was doing above with more lines of code in this script
    ##
    assign_key_to_dict_from_dict_safely(model_config, 'out_dim_override', finetune_config, 2, lprint)
    ##
    assign_key_to_dict_from_dict_safely(finetune_config, 'val_data_path', finetune_config, None, lprint)
    assign_key_to_dict_from_dict_safely(finetune_config, 'val_label_path', finetune_config, None, lprint)
    assign_key_to_dict_from_dict_safely(finetune_config, 'val_data_path_ecg', finetune_config, None, lprint)
    ##
    assign_key_to_dict_from_dict_safely(finetune_config, 'save_path_suffix', finetune_config, None, lprint)
    save_path_suffix = finetune_config['save_path_suffix']
    ##
    assign_key_to_dict_from_dict_safely(model_config, 'pooling_fxn', finetune_config, 'linear', lprint)

    ## modify train config?

    ## load model and objectives

    #
    model_name = f"gpt_{PARAMS}_ft_{finetune_config['tune_mode']}_{finetune_config['task_name']}"
    save_path = f'ft_out/{model_name}'
    if save_path_suffix:
        assert type(save_path_suffix) is str
        assert is_simple_string(save_path_suffix), f'{save_path_suffix=}'
        save_path += save_path_suffix

    ## training
    if not eval_only:
        lprint(f'final model_config: {model_config}')
        model = GPT_with_linearOutput(model_config)
        lprint(model)
        model = model.to(device)
        train_sampler = None
        if task_type == 'classification':
            train_sampler_weights = torch.DoubleTensor(make_weights_for_balanced_classes(np.load(train_label_path).astype(np.int64), nclasses=model_config['out_classes']))
            train_sampler = WeightedRandomSampler(train_sampler_weights, len(train_sampler_weights))
        objs = load_train_objs(model, train_config, model_config, finetune_config['train_data_path'], train_label_path, \
                                train_sampler=train_sampler, train_data_path_ecg=train_data_path_ecg, \
                                val_dataset_path=finetune_config['val_data_path'], val_labels_dataset_path=finetune_config['val_label_path'], val_data_path_ecg=finetune_config['val_data_path_ecg'])
        model, train_dataloader, val_dataloader, model_config, train_config, optim, lr_scheduler = objs
        model = torch.compile(model)
        count_parameters(model)
        trainer = load_trainer(model, optim, train_dataloader, save_path, train_config, lr_scheduler, model_config, None, val_loader=val_dataloader, task_type=task_type)
        lprint("Training Start")
        trainer.train(train_config["epochs"], train_config['save_freq'])
        lprint("Training End")

    ## evaluation
    model = GPT_with_linearOutput(model_config)
    lprint(model)
    ckpt_path = get_latest_file('./', pattern=f'{save_path}*.pth')
    ckpt_path_best = get_latest_file('./', pattern=f'{save_path}*best.pth')
    assert ckpt_path, f'{ckpt_path=}'
    if ckpt_path_best:
        ckpt_paths = [ckpt_path, ckpt_path_best]
    else:
        ckpt_paths = [ckpt_path]
    for ckpt_path_curr in ckpt_paths:
        load_state_dict_gpt_custom(model, ckpt_path_curr)
        _ = model.eval()
        model = model.to(device)

        # Load PPG data
        x_ppg = np.load(finetune_config['test_data_path'])

        # Load ECG data if configured
        x_ecg = None
        if model_config.get('ecg_input', False):
            x_ecg = np.load(finetune_config['test_data_path_ecg'])
            assert x_ppg.shape[0] == x_ecg.shape[0], "PPG and ECG must have same number of samples"

        if task_type == 'regression':
            # Label loading remains the same
            if isinstance(test_label_path, list):
                y_list = [np.load(path) for path in test_label_path]
                y = np.stack(y_list, axis=1)
            else:
                y = np.load(test_label_path)

            batch_size = 256
            preds_list = []
            batch_times = []

            for i in tqdm.tqdm(range(0, x_ppg.shape[0], batch_size)):
                # Process PPG batch
                xx_ppg_batch = x_ppg[i:i+batch_size, :]
                if xx_ppg_batch.ndim == 2:
                    xx_ppg_batch = xx_ppg_batch.reshape((xx_ppg_batch.shape[0], xx_ppg_batch.shape[1]//40, 40))
                elif xx_ppg_batch.ndim == 3:
                    xx_ppg_batch = xx_ppg_batch.reshape((xx_ppg_batch.shape[0], xx_ppg_batch.shape[1], xx_ppg_batch.shape[2]//40, 40))
                ppg_tensor = torch.Tensor(xx_ppg_batch).to(device)

                # Process ECG batch if available
                ecg_tensor = None
                if x_ecg is not None:
                    xx_ecg_batch = x_ecg[i:i+batch_size, :]
                    if xx_ecg_batch.ndim == 2:
                        xx_ecg_batch = xx_ecg_batch.reshape((xx_ecg_batch.shape[0], xx_ecg_batch.shape[1]//ecg_patch_size, ecg_patch_size))
                    elif xx_ecg_batch.ndim == 3:
                        xx_ecg_batch = xx_ecg_batch.reshape((xx_ecg_batch.shape[0], xx_ecg_batch.shape[1], xx_ecg_batch.shape[2]//ecg_patch_size, ecg_patch_size))
                    ecg_tensor = torch.Tensor(xx_ecg_batch).to(device)

                # 记录推理时间
                torch.cuda.synchronize()
                start_time = time.time()
                
                # Forward pass
                with torch.no_grad():
                    if ecg_tensor is not None:
                        pred_batch = model((ppg_tensor, ecg_tensor)).cpu().numpy()
                    else:
                        pred_batch = model(ppg_tensor).cpu().numpy()
                    preds_list.append(pred_batch)
                
                torch.cuda.synchronize()
                batch_time = time.time() - start_time
                batch_times.append(batch_time)

            # Rest of regression code remains the same
            preds = np.concatenate(preds_list, axis=0)
            avg_batch_time = np.mean(batch_times)
            lprint(f'平均测试batch推理时间 (回归任务): {avg_batch_time:.4f}s')

            # Determine if we are in multi-target mode by checking prediction dimensions.
            # (Even if a single target is predicted, we allow a second dimension of size 1.)
            if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
                # Single-target regression (or predictions have one neuron).
                if preds.ndim == 2:
                    preds = preds.flatten()  # Convert (N, 1) to (N,)
                valid_mask = ~np.isnan(y) & ~np.isnan(preds)
                num_invalid = np.size(y) - np.count_nonzero(valid_mask)
                if num_invalid > 0:
                    lprint(f"Found {num_invalid} (out of {np.size(y)}) samples with NaN values. Excluding these from MSE and MAE.")
                y_valid = y[valid_mask]
                preds_valid = preds[valid_mask]
                if y_valid.size > 0:
                    mse = mean_squared_error(y_valid, preds_valid)
                    mae = mean_absolute_error(y_valid, preds_valid)
                    lprint(f"\nRESULTS Total Samples: {len(y)}")
                    lprint(f"MSE: {mse}")
                    lprint(f"MAE: {mae}")
                else:
                    lprint("No valid samples available to compute MSE and MAE.")

                # Baseline using mean prediction.
                preds_mean = np.mean(y) * np.ones_like(y)
                valid_mask = ~np.isnan(y) & ~np.isnan(preds_mean)
                num_invalid = np.size(y) - np.count_nonzero(valid_mask)
                if num_invalid > 0:
                    lprint(f"Baseline: Found {num_invalid} samples with NaN values. Excluding these from MSE and MAE.")
                y_valid = y[valid_mask]
                preds_valid = preds_mean[valid_mask]
                if y_valid.size > 0:
                    mse = mean_squared_error(y_valid, preds_valid)
                    mae = mean_absolute_error(y_valid, preds_valid)
                    lprint(f"\nRESULTS WITH MEAN PREDICTION BASELINE Total Samples: {len(y)}")
                    lprint(f"MSE: {mse}")
                    lprint(f"MAE: {mae}")
                else:
                    lprint("No valid samples available for baseline computation.")
            else:
                # Multi-target regression: y and preds should have shape (N, num_targets)
                num_targets = preds.shape[1]
                for target in range(num_targets):
                    y_target = y[:, target]
                    preds_target = preds[:, target]
                    valid_mask = ~np.isnan(y_target) & ~np.isnan(preds_target)
                    num_invalid = np.size(y_target) - np.count_nonzero(valid_mask)
                    if num_invalid > 0:
                        lprint(f"Target {target}: Found {num_invalid} (out of {np.size(y_target)}) samples with NaN values. Excluding these.")
                    y_valid = y_target[valid_mask]
                    preds_valid = preds_target[valid_mask]
                    if y_valid.size > 0:
                        mse = mean_squared_error(y_valid, preds_valid)
                        mae = mean_absolute_error(y_valid, preds_valid)
                        lprint(f"\nRESULTS for target {target} - Total Samples: {len(y_target)}")
                        lprint(f"MSE: {mse}")
                        lprint(f"MAE: {mae}")
                    else:
                        lprint(f"Target {target}: No valid samples available for computing MSE and MAE.")

                # Baseline with mean prediction for each target.
                preds_mean = np.mean(y, axis=0)  # shape: (num_targets,)
                preds_mean = np.tile(preds_mean, (y.shape[0], 1))
                for target in range(num_targets):
                    y_target = y[:, target]
                    preds_target = preds_mean[:, target]
                    valid_mask = ~np.isnan(y_target) & ~np.isnan(preds_target)
                    num_invalid = np.size(y_target) - np.count_nonzero(valid_mask)
                    if num_invalid > 0:
                        lprint(f"Baseline Target {target}: Found {num_invalid} samples with NaN values. Excluding these.")
                    y_valid = y_target[valid_mask]
                    preds_valid = preds_target[valid_mask]
                    if y_valid.size > 0:
                        mse = mean_squared_error(y_valid, preds_valid)
                        mae = mean_absolute_error(y_valid, preds_valid)
                        lprint(f"\nRESULTS WITH MEAN PREDICTION BASELINE for target {target} - Total Samples: {len(y_target)}")
                        lprint(f"MSE: {mse}")
                        lprint(f"MAE: {mae}")
                    else:
                        lprint(f"Baseline Target {target}: No valid samples available for computing MSE and MAE.")

            # Save predictions.
            y_pred_save_path = '/'.join(finetune_config['test_data_path'].split('/')[:-1] + ['preds.npy'])
            np.save(y_pred_save_path, preds)
            lprint(f'SAVED: {y_pred_save_path}')

        # Assuming task_type, model, device, fs, x, y are defined above
        elif task_type == 'classification':
            y = np.load(test_label_path)
            batch_size = 256
            preds = []
            batch_times = []

            for i in tqdm.tqdm(range(0, x_ppg.shape[0], batch_size)):
                # Process PPG batch
                xx_ppg_batch = x_ppg[i:i+batch_size, :]
                if xx_ppg_batch.ndim == 2:
                    xx_ppg_batch = xx_ppg_batch.reshape((xx_ppg_batch.shape[0], xx_ppg_batch.shape[1]//40, 40))
                elif xx_ppg_batch.ndim == 3:
                    xx_ppg_batch = xx_ppg_batch.reshape((xx_ppg_batch.shape[0], xx_ppg_batch.shape[1], xx_ppg_batch.shape[2]//40, 40))
                ppg_tensor = torch.Tensor(xx_ppg_batch).to(device)

                # Process ECG batch if available
                ecg_tensor = None
                if x_ecg is not None:
                    xx_ecg_batch = x_ecg[i:i+batch_size, :]
                    if xx_ecg_batch.ndim == 2:
                        xx_ecg_batch = xx_ecg_batch.reshape((xx_ecg_batch.shape[0], xx_ecg_batch.shape[1]//ecg_patch_size, ecg_patch_size))
                    elif xx_ecg_batch.ndim == 3:
                        xx_ecg_batch = xx_ecg_batch.reshape((xx_ecg_batch.shape[0], xx_ecg_batch.shape[1], xx_ecg_batch.shape[2]//ecg_patch_size, ecg_patch_size))
                    ecg_tensor = torch.Tensor(xx_ecg_batch).to(device)

                # 记录推理时间
                torch.cuda.synchronize()
                start_time = time.time()

                # Forward pass
                with torch.no_grad():
                    if ecg_tensor is not None:
                        logits = model((ppg_tensor, ecg_tensor))
                    else:
                        logits = model(ppg_tensor)
                    pred_batch = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.extend(pred_batch.tolist())
                
                torch.cuda.synchronize()
                batch_time = time.time() - start_time
                batch_times.append(batch_time)

            # Rest of classification code remains the same
            avg_batch_time = np.mean(batch_times)
            lprint(f'平均测试batch推理时间 (分类任务): {avg_batch_time:.4f}s')
            
            y = np.array(y)
            preds = np.array(preds)
            valid_mask = ~np.isnan(y) & ~np.isnan(preds)
            num_invalid = np.size(y) - np.count_nonzero(valid_mask)
            if num_invalid > 0:
                lprint(f"Found {num_invalid} samples with NaN values. These will be excluded from metric calculations.")

            y_valid = y[valid_mask]
            preds_valid = preds[valid_mask]

            if y_valid.size > 0:
                # Determine if binary or multiclass
                unique_classes = np.unique(y_valid)
                num_classes = len(unique_classes)
                is_binary = (num_classes == 2)

                # For binary classification, use 'binary' for F1; for multiclass, use 'macro'
                f1_average = 'binary' if is_binary else 'macro'

                # Directly use preds_labels as class predictions
                preds_labels = preds_valid.astype(int)

                # Compute basic metrics
                f1_normal = f1_score(y_valid, preds_labels, average=f1_average)
                acc_normal = accuracy_score(y_valid, preds_labels)
                mcc_normal = matthews_corrcoef(y_valid, preds_labels)

                # AUROC and AUPRC require probability estimates for meaningful results.
                # Since we only have class predictions, we cannot compute them in a standard way.
                # We will skip them or set them to NaN.
                if is_binary:
                    # Attempt to compute AUPRC and AUROC with only class labels (degenerate scenario)
                    # This will produce a precision_recall_curve with a single threshold point.
                    try:
                        precision_normal, recall_normal, _ = precision_recall_curve(y_valid, preds_labels)
                        auprc_normal = auc(recall_normal, precision_normal)
                    except:
                        auprc_normal = float('nan')

                    # For AUROC with only class labels (no probabilities), roc_auc_score won't be meaningful:
                    # It will be equivalent to accuracy if only 0/1 predictions are given.
                    # We can still attempt it:
                    try:
                        auroc_normal = roc_auc_score(y_valid, preds_labels) if is_binary else float('nan')
                    except:
                        auroc_normal = float('nan')
                else:
                    # Multiclass scenario:
                    # Without probabilities, we cannot compute meaningful AUPRC or AUROC.
                    auprc_normal = float('nan')
                    auroc_normal = float('nan')

                # Compute Majority Class Baseline
                majority_class = np.bincount(y_valid.astype(int)).argmax()
                preds_majority = np.full_like(y_valid, majority_class)
                acc_majority = accuracy_score(y_valid, preds_majority)
                f1_majority = f1_score(y_valid, preds_majority, zero_division=0, average=f1_average)
                mcc_majority = matthews_corrcoef(y_valid, preds_majority)

                # For AUPRC and AUROC in majority baseline:
                if is_binary:
                    # Degenerate scenario again, using only class labels
                    try:
                        precision_majority, recall_majority, _ = precision_recall_curve(y_valid, preds_majority)
                        auprc_majority = auc(recall_majority, precision_majority)
                    except:
                        auprc_majority = float('nan')
                    try:
                        auroc_majority = roc_auc_score(y_valid, preds_majority)
                    except:
                        auroc_majority = float('nan')
                else:
                    auprc_majority = float('nan')
                    auroc_majority = float('nan')

                # Random Predictions Baseline
                np.random.seed(42)  # For reproducibility
                # Probability of class 1 = prevalence in y_valid (only meaningful if binary)
                if is_binary:
                    prob_class_1 = np.mean(y_valid == 1)
                    preds_random = np.random.choice([0, 1], size=len(y_valid), p=[1 - prob_class_1, prob_class_1])
                else:
                    # For multiclass, choose classes uniformly at random:
                    preds_random = np.random.choice(unique_classes, size=len(y_valid))

                acc_random = accuracy_score(y_valid, preds_random)
                f1_random = f1_score(y_valid, preds_random, zero_division=0, average=f1_average)
                mcc_random = matthews_corrcoef(y_valid, preds_random)

                if is_binary:
                    try:
                        precision_random, recall_random, _ = precision_recall_curve(y_valid, preds_random)
                        auprc_random = auc(recall_random, precision_random)
                    except:
                        auprc_random = float('nan')
                    try:
                        auroc_random = roc_auc_score(y_valid, preds_random)
                    except:
                        auroc_random = float('nan')
                else:
                    auprc_random = float('nan')
                    auroc_random = float('nan')

                # Print results
                lprint(f"\nTotal Samples: {len(y)}")
                lprint(f"Valid Samples: {y_valid.size}\n")

                lprint("=== Normal Predictions (Model's Predictions) ===")
                lprint(f"Accuracy: {acc_normal:.4f}")
                lprint(f"F1 Score ({f1_average}): {f1_normal:.4f}")
                lprint(f"AUPRC: {auprc_normal if not np.isnan(auprc_normal) else 'N/A'}")
                lprint(f"MCC: {mcc_normal:.4f}")
                lprint(f"AUROC: {auroc_normal if not np.isnan(auroc_normal) else 'N/A'}\n")

                lprint(f"=== Majority Class Baseline (Predicting Class: {majority_class}) ===")
                lprint(f"Accuracy: {acc_majority:.4f}")
                lprint(f"F1 Score ({f1_average}): {f1_majority:.4f}")
                lprint(f"AUPRC: {auprc_majority if not np.isnan(auprc_majority) else 'N/A'}")
                lprint(f"MCC: {mcc_majority:.4f}")
                lprint(f"AUROC: {auroc_majority if not np.isnan(auroc_majority) else 'N/A'}\n")

                lprint("=== Random Predictions Baseline ===")
                lprint(f"Accuracy: {acc_random:.4f}")
                lprint(f"F1 Score ({f1_average}): {f1_random:.4f}")
                lprint(f"AUPRC: {auprc_random if not np.isnan(auprc_random) else 'N/A'}")
                lprint(f"MCC: {mcc_random:.4f}")
                lprint(f"AUROC: {auroc_random if not np.isnan(auroc_random) else 'N/A'}\n")

            y_pred_save_path = '/'.join(finetune_config['test_data_path'].split('/')[:-1] + ['preds.npy'])
            np.save(y_pred_save_path, preds)
            lprint(f'SAVED: {y_pred_save_path}')

        else:
            raise NotImplementedError(f'{task_type}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_config_path", type=str, required=True)
    parser.add_argument("--eval_only", action="store_true", help="Enable evaluation only mode")
    parser.add_argument("--train_only", action="store_true", help="Enable train only mode")
    args = parser.parse_args()

    finetune_config_path = args.finetune_config_path
    eval_only = args.eval_only
    train_only = args.train_only
    
    do_train()
