#!/usr/bin/env python
import sys
sys.path.append('.')
import os
import numpy as np
import re
import json
import logging
logging.basicConfig(level=logging.INFO)
from logging import info as lprint
from safetensors.torch import load_file as load_file_st
from peft import get_peft_model, LoraConfig, TaskType, AdaLoraConfig

import torch
import torch.nn as nn

from model.embedding import *
from model.attention import *

from local.supp_fxns import *
from local.feat_fxns import *
from model.sqi_net import SQINet
from model.acc_net import ACCNet
# from model.ppg2ecgmetrics_net import PPG2ECGmetricsNet


def load_state_dict_gpt(model, state_dict_path, strict=True, patch_size=40):
    out_dim = 2 * patch_size
    assert os.path.isfile(state_dict_path), f'{state_dict_path=}'
    if state_dict_path.endswith('.pth'):    # david code compatibility
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))['model']
        for key in list(state_dict):
            if key.startswith('module.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        for key in list(state_dict):
            if key.startswith('_orig_mod.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        for key in list(state_dict):
            if key.startswith('gpt.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        result = model.load_state_dict(state_dict, strict=strict)
        loaded_percentage = ((len(model.state_dict()) - len(result.missing_keys)) / len(model.state_dict())) * 100
        lprint(f'{loaded_percentage=} for {state_dict_path=}')
    elif state_dict_path.endswith('.safetensors'):  # jiaying code compatibility
        state_dict = load_file_st(state_dict_path)
        for key in list(state_dict):
            if key.startswith('_orig_mod.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=strict)
    elif state_dict_path.endswith('.pt'):   # saurabh code compatibility
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'), weights_only=False)
        state_dict = state_dict['model_state_dict']
        for key in list(state_dict):
            if key.startswith('_orig_mod.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise NotImplementedError(f'{state_dict_path=} EXTENSION UNSUPPORTED')
    print(f'LOADED: {state_dict_path}')


def unfreeze_selected_layers(model):
    # Freeze entire GPT first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze last 3 GPT layers
    for layer in model.gpt_layers[-6:]:  # Negative indexing gets last 3
        for param in layer.parameters():
            param.requires_grad = True


class GPT_with_linearOutput(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        if model_config['model_name'] == 'gpt':
            self.gpt_class = GPT
            self.gpt = self.gpt_class(**model_config)
        elif model_config['model_name'] in ['accxgpt', 'accygpt', 'acczgpt', 'sqigpt', 'accgpt', 'sqiaccgpt']:
            if model_config['model_name'] == 'accxgpt':
                self.gpt_class = ACCX_GPT
            elif model_config['model_name'] == 'accygpt':
                self.gpt_class = ACCY_GPT
            elif model_config['model_name'] == 'acczgpt':
                self.gpt_class = ACCZ_GPT
            elif model_config['model_name'] == 'sqigpt':
                self.gpt_class = SQI_GPT
            elif model_config['model_name'] == 'accgpt':
                self.gpt_class = ACC_GPT
            elif model_config['model_name'] == 'sqiaccgpt':
                self.gpt_class = SQIACC_GPT
            else:
                raise
            self.gpt = self.gpt_class(model_config)
        else:
            raise NotImplementedError(f"{model_config['model_name']=}")
        self.PARAMS = model_config['PARAMS']
        if model_config['model_name'] == 'gpt':
            if self.PARAMS == '19M': # gpt_19M_49.pth ; gpt_19M_ep19_cauchy.pt ; gpt_19M_ep19_studentt.pt ; gpt_19M_ep19_studentt_adaptive.pt
                state_dict_path = 'gpt_19M_ep19_laplace.pt' #'gpt_19M_49.pth'
            else:
                state_dict_path = None
            if state_dict_path:
                state_dict_path = './gpt_base_models/' + state_dict_path
        else:
            state_dict_path = model_config['gpt_state_dict_path']
        
        # 只有在有预训练权重路径时才加载
        if state_dict_path:
            load_state_dict_gpt(self.gpt, state_dict_path)
        else:
            lprint(f"跳过预训练权重加载，Student模型 {self.PARAMS} 将从头开始训练")
        self.tune_mode = model_config['tune_mode']
        if self.tune_mode == 'head':
            freeze_nn(self.gpt)
        elif self.tune_mode == 'full':
            unfreeze_nn(self.gpt)
#            unfreeze_selected_layers(self.gpt)
        else:
            raise NotImplementedError(f'{self.tune_mode=}')
        self.d_model = model_config['d_model']
#        self.norm1 = nn.LayerNorm(self.d_model)
#        self.act1 = nn.ReLU()
        self.act1 = nn.GELU()
#        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.n_patches = model_config['n_patches']
        self.out_classes = model_config['out_classes']
        self.apply_mask = model_config['apply_mask']
        self.use_penultimate_layer = model_config['use_penultimate_layer']
        self.pooling_fxn = model_config['pooling_fxn']
        bottleneck_layer_dim = 128
        assert not (self.use_penultimate_layer and self.pooling_fxn != 'linear'), f'penultimate layer option only allowed in linear pooling for now {self.use_penultimate_layer} {self.pooling_fxn=}'
        if self.pooling_fxn == 'linear':
            if self.use_penultimate_layer:
                self.linear = nn.Linear((self.n_patches+1)*self.d_model, bottleneck_layer_dim, bias=True)       # +1 because of the special token in GPT which we don't drop
                self.norm2 = nn.LayerNorm(bottleneck_layer_dim)
                self.linear2 = nn.Linear(bottleneck_layer_dim, self.out_classes, bias=True)
            else:
                self.linear = nn.Linear((self.n_patches+1)*self.d_model, self.out_classes, bias=True)
            out_dim_pool = self.n_patches+1
        elif self.pooling_fxn == 'attn':
            input_dim = self.d_model
            self.attn_pooler = AttentionPooling(input_dim)
            self.linear = nn.Linear(1*self.d_model, self.out_classes, bias=True)
            out_dim_pool = 1
        elif self.pooling_fxn == 'lde':
            num_codewords = 4
            input_dim = self.d_model
            self.lde_pooler = LDEPooling(input_dim, num_centers=num_codewords)  # B,S,D -> B,D*K
            self.linear = nn.Linear(num_codewords*self.d_model, self.out_classes)
            out_dim_pool = num_codewords
        elif self.pooling_fxn == 'quartile':
            q_points=[0.1, 0.25, 0.5, 0.75, 0.9]
            self.quartile_pooler = QuartilePooling(q_points)
            self.linear = nn.Linear(len(q_points)*self.d_model, self.out_classes)
            out_dim_pool = len(q_points)
        elif self.pooling_fxn == 'asp':
            self.asp_pooler = AttentiveStatsPool(self.d_model)
            self.linear = nn.Linear(2*self.d_model, self.out_classes)
            out_dim_pool = 2
        elif self.pooling_fxn == 'universal':
            self.universal_pooler = EnhancedUniversalPool(self.d_model)
            self.linear = nn.Linear(2*self.d_model, self.out_classes)
            out_dim_pool = 2
        elif self.pooling_fxn == 'lastminmax':
            self.lastminmax_pooler = LastMinMaxPool()
            self.linear = nn.Linear(3*self.d_model, self.out_classes)
            out_dim_pool = 3
        else:
            raise NotImplementedError(f'{self.pooling_fxn=}')
        self.norm1 = nn.LayerNorm(out_dim_pool * self.d_model)
        #
        if self.model_config['ecg_input']:
            self.ecg_patch_embedding = None
            ecg_patch_size = 40
#            self.ecg_patch_embedding = PatchEmbedding(ecg_patch_size, self.d_model)
            self.ppg_ecg_patch_emb_fuser = nn.Linear(self.d_model*2, self.d_model)      #nn.Linear(self.d_model*2, self.d_model) , GatedFusion(self.d_model)
        else:
            self.ecg_patch_embedding = self.ppg_ecg_patch_emb_fuser = None
        # fuse tuning related stuff
        self.fuse_tuning = model_config['fuse_tuning']
        self.fuse_feat_type = model_config['fuse_feat_type']
        if self.fuse_tuning:
            if self.fuse_feat_type == 'ppg2ecgmetrics':
                self.fuse_feat_extractor = PPG2ECGmetricsFuser(self.d_model)
            elif self.fuse_feat_type == 'stft':
                self.fuse_feat_extractor = SpectralPhaseFeatureExtractor(40, self.d_model, use_e=True, normalize=True, fusion_method='gated')
            elif self.fuse_feat_type == 'mrstft':
                self.fuse_feat_extractor = MultiScaleSTFTPhaseFeatureExtractor(d_model=self.d_model, use_e=True, normalize=True, learnable_window=False, fusion_method='gated')
            elif self.fuse_feat_type == 'moment':
                self.fuse_feat_extractor =  MomentFeatureExtractor(d_model=self.d_model, use_e=True, normalize=True, fusion_method='gated', model_name='AutonLab/MOMENT-1-base')
            elif self.fuse_feat_type == 'pii':
                self.fuse_feat_extractor = PIIFeatureExtractor(d_model=self.d_model, use_e=True, normalize=True, fusion_method='gated')
            else:
                raise NotImplementedError(f'{self.fuse_feat_type=}')
        else:
            self.fuse_feat_extractor = None
        # lora
        self.use_lora = model_config['use_lora']
        if self.use_lora:
            peft_config = LoraConfig(r=8, lora_alpha=16, target_modules={'w_q', 'w_v'})     # scale in LoRA is defined as alpha/rank; dont tune alpha
#            peft_config = AdaLoraConfig(peft_type="ADALORA", init_r=12, target_r=8, lora_alpha=16, target_modules={'w_q', 'w_v'})     # scale in LoRA is defined as alpha/rank; dont tune alpha
            print('GPT before lora', self.gpt)
            self.gpt = get_peft_model(self.gpt, peft_config)
            print('GPT after lora', self.gpt)
#            if self.use_conditional_lora:
#                self.use_sequence = False
#                self.use_learnable_component = True
#                replace_lora_with_conditional_lora(self.gpt, layer_indices=None, lora_rank=4, lora_alpha=32, dropout=0, stft_dim=d_model, use_sequence=self.use_sequence, use_learnable_component=self.use_learnable_component, verbose=True)

    def forward(self, x, get_gpt_output=False):
        # fwd pass through base GPT model
        if self.model_config['model_name'] == 'gpt':
            x_all = self.gpt.encode(x, apply_mask=self.apply_mask, return_all_encodings=True, fuse_feat_extractor=self.fuse_feat_extractor, \
                                ecg_input=self.model_config['ecg_input'], ecg_patch_embedding=self.ecg_patch_embedding, ppg_ecg_patch_emb_fuser=self.ppg_ecg_patch_emb_fuser)
            x = x_all[-1]
            if get_gpt_output:
                x2 = self.gpt(x, apply_mask=self.apply_mask, do_encode=False)
        elif self.model_config['model_name'] in ['sqigpt', 'accxgpt', 'accygpt', 'acczgpt', 'accgpt', 'sqiaccgpt']:
            xe, xo = self.gpt(x, apply_mask=self.apply_mask, get_gpt_output=True)
            if get_gpt_output:
                x2 = xo
            x = xe
#            raise Exception(f'c2 {x.shape=}')   # x.shape=torch.Size([32, 3, 512])
        else:
            raise NotImplementedError(f"{self.model_config['model_name']=}")
        #
#        x = self.norm1(x)
#        x = self.act1(x)
        if self.pooling_fxn == 'linear':
            x = x.reshape(x.shape[0], -1)
        elif self.pooling_fxn == 'attn':
            x = self.attn_pooler(x)
        elif self.pooling_fxn == 'lde':
            x = self.lde_pooler(x)
        elif self.pooling_fxn == 'quartile':
            x = self.quartile_pooler(x)
        elif self.pooling_fxn == 'asp':
            x = self.asp_pooler(x)
        elif self.pooling_fxn == 'universal':
            x_all = [self.norm1(xx) for xx in x_all]
            x_all = [self.act1(xx) for xx in x_all]
            x = self.universal_pooler(x_all)
        elif self.pooling_fxn == 'lastminmax':
            x = self.lastminmax_pooler(x)
        else:
            raise NotImplementedError(f'{self.pooling_fxn=}')
        x = self.norm1(x)
        x = self.act1(x)
        x = self.linear(x)

        if self.use_penultimate_layer:
            x = self.norm2(x)
            x = self.act1(x)
            x = self.linear2(x)

        if get_gpt_output:
#            lprint(f'{x.shape=} {x2.shape=}'); raise Exception
            return (x, x2)
        else:
            return x


class SQIACC_GPT(nn.Module):
    def __init__(self, gpt_model_config):
        super().__init__()
        # GPT
        self.gpt = GPTv2(**gpt_model_config)

        # SQINet
        self.sqi_net = SQINet()
        sqi_state_dict_path = './pretrained_models/sqi_best_model.pth'
        assert os.path.isfile(sqi_state_dict_path), f'{sqi_state_dict_path=}'
        logging.info(f'LOADING: {sqi_state_dict_path=}')
        sqi_state_dict = torch.load(sqi_state_dict_path, map_location=torch.device('cpu'))
        self.sqi_net.load_state_dict(sqi_state_dict)
        freeze_nn(self.sqi_net)

        # ACCNet
        self.acc_net = ACCNet()
        acc_state_dict_path = './pretrained_models/ppg_to_acc_gan_generator_v7_2.pth'
        assert os.path.isfile(acc_state_dict_path), f'{acc_state_dict_path=}'
        logging.info(f'LOADING: {acc_state_dict_path=}')
        acc_state_dict = torch.load(acc_state_dict_path, map_location=torch.device('cpu'))
        self.acc_net.load_state_dict(acc_state_dict)
        freeze_nn(self.acc_net)

        # fuse ACC and GPT
        self.d_model = gpt_model_config['d_model']
        self.patch_size = gpt_model_config['patch_size']
        self.sqi_patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.acc_patch_embeddings = nn.ModuleList([PatchEmbedding(self.patch_size, self.d_model) for _ in range(3)])
        self.fuser = MultiGatedFusion(self.d_model, num_sources=5)

    def forward(self, x, apply_mask=True, get_gpt_output=False):   # x.shape = B, S, P
        B, N, P = x.shape
        ppg_patch_embedding = self.gpt.encode(x, apply_mask, return_patch_embedding=True)     # this just calculates patch embedding - B, S+1, d_model
        #
        sqi_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        sqi_signal = self.sqi_net(sqi_signal)   # B,L
        sqi_signal = sqi_signal.reshape(B, N, P)
        sqi_patch_embedding = self.sqi_patch_embedding(sqi_signal)
        #
        acc_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        L = acc_signal.shape[-1]
        acc_signal = acc_signal.unsqueeze(1)    # B,1,L
        acc_signal = F.interpolate(acc_signal, size=int(L * 128 / 40), mode='linear', align_corners=False)  # B,1,L2
        acc_signal = self.acc_net(acc_signal)   # B,3,L2
        acc_signal = F.interpolate(acc_signal, size=L, mode='linear', align_corners=False)  # B,3,L
        acc_patch_embedding = []
        for i in range(3):
            a0 = acc_signal[:,i,:]
            a0 = a0.reshape(B, N, P)
            a0 = self.acc_patch_embeddings[i](a0)
            acc_patch_embedding.append(a0)
        #
        x = self.fuser([ppg_patch_embedding] + [sqi_patch_embedding] + acc_patch_embedding)
        #
        x = self.gpt.encode(x, apply_mask, skip_patch_embedding=True)
        xe = x
        x = self.gpt(xe, apply_mask, skip_encoding=True)
        if get_gpt_output:
            return xe, x
        else:
            return x

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class ACC_GPT(nn.Module):
    def __init__(self, gpt_model_config):
        super().__init__()
        # GPT
        self.gpt = GPTv2(**gpt_model_config)

        # ACCNet
        self.acc_net = ACCNet()
        acc_state_dict_path = './pretrained_models/ppg_to_acc_gan_generator_v7_2.pth'
        assert os.path.isfile(acc_state_dict_path), f'{acc_state_dict_path=}'
        logging.info(f'LOADING: {acc_state_dict_path=}')
        acc_state_dict = torch.load(acc_state_dict_path, map_location=torch.device('cpu'))
        self.acc_net.load_state_dict(acc_state_dict)
        freeze_nn(self.acc_net)

        # fuse ACC and GPT
        self.d_model = gpt_model_config['d_model']
        self.patch_size = gpt_model_config['patch_size']
        self.acc_patch_embeddings = nn.ModuleList([PatchEmbedding(self.patch_size, self.d_model) for _ in range(3)])    # nn.ModuleList is important, cant have just list, pytorch cant register parameters properly
        self.fuser = MultiGatedFusion(self.d_model, num_sources=4)

    def forward(self, x, apply_mask=True, get_gpt_output=False):   # x.shape = B, S, P
        B, N, P = x.shape
        ppg_patch_embedding = self.gpt.encode(x, apply_mask, return_patch_embedding=True)     # this just calculates patch embedding - B, S+1, d_model
        acc_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        L = acc_signal.shape[-1]
        acc_signal = acc_signal.unsqueeze(1)    # B,1,L
        acc_signal = F.interpolate(acc_signal, size=int(L * 128 / 40), mode='linear', align_corners=False)  # B,1,L2
        acc_signal = self.acc_net(acc_signal)   # B,3,L2
        acc_signal = F.interpolate(acc_signal, size=L, mode='linear', align_corners=False)  # B,3,L
        acc_patch_embedding = []
        for i in range(3):
            a0 = acc_signal[:,i,:]
            a0 = a0.reshape(B, N, P)
            a0 = self.acc_patch_embeddings[i](a0)
            acc_patch_embedding.append(a0)
        x = self.fuser([ppg_patch_embedding] + acc_patch_embedding)
        x = self.gpt.encode(x, apply_mask, skip_patch_embedding=True)
        xe = x
        x = self.gpt(xe, apply_mask, skip_encoding=True)
        if get_gpt_output:
            return xe, x
        else:
            return x

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class SQI_GPT(nn.Module):
    def __init__(self, gpt_model_config):
        super().__init__()
        # GPT
        self.gpt = GPTv2(**gpt_model_config)

        # SQINet
        self.sqi_net = SQINet()
        sqi_state_dict_path = './pretrained_models/sqi_best_model.pth'
        assert os.path.isfile(sqi_state_dict_path), f'{sqi_state_dict_path=}'
        logging.info(f'LOADING: {sqi_state_dict_path=}')
        sqi_state_dict = torch.load(sqi_state_dict_path, map_location=torch.device('cpu'))
        self.sqi_net.load_state_dict(sqi_state_dict)
        freeze_nn(self.sqi_net)

        # fuse SQI and GPT
        self.d_model = gpt_model_config['d_model']
        self.patch_size = gpt_model_config['patch_size']
        self.sqi_patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.fuser = GatedFusion(self.d_model)

    def forward(self, x, apply_mask=True, get_gpt_output=False):   # x.shape = B, S, P
        B, N, P = x.shape
        ppg_patch_embedding = self.gpt.encode(x, apply_mask, return_patch_embedding=True)     # this just calculates patch embedding - B, S+1, d_model
        sqi_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        sqi_signal = self.sqi_net(sqi_signal)   # B,L
        sqi_signal = sqi_signal.reshape(B, N, P)
        sqi_patch_embedding = self.sqi_patch_embedding(sqi_signal)
        x = self.fuser(ppg_patch_embedding, sqi_patch_embedding)
        x = self.gpt.encode(x, apply_mask, skip_patch_embedding=True)
        xe = x
        x = self.gpt(xe, apply_mask, skip_encoding=True)
        if get_gpt_output:
            return xe, x
        else:
            return x

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class ACCZ_GPT(nn.Module):
    def __init__(self, gpt_model_config):
        super().__init__()
        # GPT
        self.gpt = GPTv2(**gpt_model_config)

        # ACCNet
        self.acc_net = ACCNet()
        acc_state_dict_path = './pretrained_models/ppg_to_acc_gan_generator_v7_2.pth'
        assert os.path.isfile(acc_state_dict_path), f'{acc_state_dict_path=}'
        logging.info(f'LOADING: {acc_state_dict_path=}')
        acc_state_dict = torch.load(acc_state_dict_path, map_location=torch.device('cpu'))
        self.acc_net.load_state_dict(acc_state_dict)
        freeze_nn(self.acc_net)

        # fuse ACC and GPT
        self.d_model = gpt_model_config['d_model']
        self.patch_size = gpt_model_config['patch_size']
        self.acc_patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.fuser = GatedFusion(self.d_model)

    def forward(self, x, apply_mask=True, get_gpt_output=False):   # x.shape = B, S, P
        B, N, P = x.shape
        ppg_patch_embedding = self.gpt.encode(x, apply_mask, return_patch_embedding=True)     # this just calculates patch embedding - B, S+1, d_model
        acc_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        L = acc_signal.shape[-1]
        acc_signal = acc_signal.unsqueeze(1)    # B,1,L
        acc_signal = F.interpolate(acc_signal, size=int(L * 128 / 40), mode='linear', align_corners=False)  # B,1,L2
        acc_signal = self.acc_net(acc_signal)[:,2,:].unsqueeze(1)   # B,1,L2
        acc_signal = F.interpolate(acc_signal, size=L, mode='linear', align_corners=False)  # B,1,L
        acc_signal = acc_signal.squeeze(1)    # B,L
        acc_signal = acc_signal.reshape(B, N, P)
        acc_patch_embedding = self.acc_patch_embedding(acc_signal)
        x = self.fuser(ppg_patch_embedding, acc_patch_embedding)
        x = self.gpt.encode(x, apply_mask, skip_patch_embedding=True)
        xe = x
        x = self.gpt(xe, apply_mask, skip_encoding=True)
        if get_gpt_output:
            return xe, x
        else:
            return x

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class ACCY_GPT(nn.Module):
    def __init__(self, gpt_model_config):
        super().__init__()
        # GPT
        self.gpt = GPTv2(**gpt_model_config)

        # ACCNet
        self.acc_net = ACCNet()
        acc_state_dict_path = './pretrained_models/ppg_to_acc_gan_generator_v7_2.pth'
        assert os.path.isfile(acc_state_dict_path), f'{acc_state_dict_path=}'
        logging.info(f'LOADING: {acc_state_dict_path=}')
        acc_state_dict = torch.load(acc_state_dict_path, map_location=torch.device('cpu'))
        self.acc_net.load_state_dict(acc_state_dict)
        freeze_nn(self.acc_net)

        # fuse ACC and GPT
        self.d_model = gpt_model_config['d_model']
        self.patch_size = gpt_model_config['patch_size']
        self.acc_patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.fuser = GatedFusion(self.d_model)

    def forward(self, x, apply_mask=True, get_gpt_output=False):   # x.shape = B, S, P
        B, N, P = x.shape
        ppg_patch_embedding = self.gpt.encode(x, apply_mask, return_patch_embedding=True)     # this just calculates patch embedding - B, S+1, d_model
        acc_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        L = acc_signal.shape[-1]
        acc_signal = acc_signal.unsqueeze(1)    # B,1,L
        acc_signal = F.interpolate(acc_signal, size=int(L * 128 / 40), mode='linear', align_corners=False)  # B,1,L2
        acc_signal = self.acc_net(acc_signal)[:,1,:].unsqueeze(1)   # B,1,L2
        acc_signal = F.interpolate(acc_signal, size=L, mode='linear', align_corners=False)  # B,1,L
        acc_signal = acc_signal.squeeze(1)    # B,L
        acc_signal = acc_signal.reshape(B, N, P)
        acc_patch_embedding = self.acc_patch_embedding(acc_signal)
        x = self.fuser(ppg_patch_embedding, acc_patch_embedding)
        x = self.gpt.encode(x, apply_mask, skip_patch_embedding=True)
        xe = x
        x = self.gpt(xe, apply_mask, skip_encoding=True)
        if get_gpt_output:
            return xe, x
        else:
            return x

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class ACCX_GPT(nn.Module):
    def __init__(self, gpt_model_config):
        super().__init__()
        # GPT
        self.gpt = GPTv2(**gpt_model_config)    ##

        # ACCNet
        self.acc_net = ACCNet()
        acc_state_dict_path = './pretrained_models/ppg_to_acc_gan_generator_v7_2.pth'
        assert os.path.isfile(acc_state_dict_path), f'{acc_state_dict_path=}'
        logging.info(f'LOADING: {acc_state_dict_path=}')
        acc_state_dict = torch.load(acc_state_dict_path, map_location=torch.device('cpu'))
        self.acc_net.load_state_dict(acc_state_dict)
        freeze_nn(self.acc_net)

        # fuse ACC and GPT
        self.d_model = gpt_model_config['d_model']
        self.patch_size = gpt_model_config['patch_size']
        self.acc_patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.fuser = GatedFusion(self.d_model)

    def forward(self, x, apply_mask=True, get_gpt_output=False):   # x.shape = B, S, P
        B, N, P = x.shape
        ppg_patch_embedding = self.gpt.encode(x, apply_mask, return_patch_embedding=True)     # this just calculates patch embedding - B, S+1, d_model
        acc_signal = x.reshape(B, -1)   # this should usually make this B, 1200 since we train with 30 sec i.e. 1200 samples for our 40Hz signal
        L = acc_signal.shape[-1]
        acc_signal = acc_signal.unsqueeze(1)    # B,1,L
        acc_signal = F.interpolate(acc_signal, size=int(L * 128 / 40), mode='linear', align_corners=False)  # B,1,L2
        acc_signal = self.acc_net(acc_signal)[:,0,:].unsqueeze(1)   # B,1,L2
        acc_signal = F.interpolate(acc_signal, size=L, mode='linear', align_corners=False)  # B,1,L
        acc_signal = acc_signal.squeeze(1)    # B,L
        acc_signal = acc_signal.reshape(B, N, P)
        acc_patch_embedding = self.acc_patch_embedding(acc_signal)
        x = self.fuser(ppg_patch_embedding, acc_patch_embedding)
        x = self.gpt.encode(x, apply_mask, skip_patch_embedding=True)
        xe = x  ##
        x = self.gpt(xe, apply_mask, skip_encoding=True)
        if get_gpt_output:
            return xe, x
        else:
            return x

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class GPTv2(nn.Module):
    def __init__(self, patch_size: int, d_model: int, n_heads: int, n_layers: int,
                 dropout=0.1, max_len=2400, loss="mse", with_conv=False, overlap=0, length=0, **kwargs):
        """
        GPT Model Definition
        :param loss: If "laplace", then the model outputs 2 channels, one for mu and one for b. This is used for
        logit laplace loss. If "mse", then the model simply outputs 1 channel in the input signal space
        :param with_conv: If True, a causal convolutional layer of kernel 3 stride 1 is applied before attention.
        It is causal because for k-th position, the convolution kernel is applied to k-2, k-1, and k-th position.
        This is achieved by padding 2 0-vectors in front.
        """
        super().__init__()
        print(f'x1 {loss=}')
        if kwargs:
            print(f"Ignored keyword arguments: {kwargs}")
        self.patch_size, self.d_model, self.n_head, self.n_layers = patch_size, d_model, n_heads, n_layers
        self.patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.gpt_layers = nn.ModuleList([GPTLayer(d_model=d_model,
                                                  n_heads=n_heads,
                                                  dropout=dropout,
                                                  max_len=max_len,
                                                  with_conv=with_conv)
                                         for _ in range(n_layers)])
        if loss == "mse":
            self.out_dim = patch_size
        elif loss in ["mixgaussian", "mixcauchy", "mixlaplace", "mixstudentt"]:
            self.out_dim = 3 * 3 * patch_size
        elif loss == "studentt_adaptive":
            self.out_dim = 3 * patch_size
        elif loss == "alphastable":
            self.out_dim = 4 * patch_size
        elif loss == "mixalphastable":
            self.out_dim = 5 * 3 * patch_size
        elif loss == "mixstudentt_adaptive":
            self.out_dim = 4 * 3 * patch_size
        else:
            self.out_dim = 2 * patch_size
        self.multiplier = int(self.out_dim / patch_size)
#        self.out_dim = patch_size if loss == "mse" else 2 * patch_size
        print(f'{self.out_dim=}')
        self.sigmoid = nn.Sigmoid()
        self.encoded_dim = d_model
        self.head = nn.Linear(self.d_model, self.out_dim)

    def forward(self, x: torch.Tensor, apply_mask: bool, skip_encoding=False) -> torch.Tensor:
        """
        Get the GPT output. Useful in pretraining
        :param x: input tensor
        :param apply_mask: If true, causal mask is applied. Set this flag to true in training, and false in inference
        :return: GPT predictions
        """
        if not skip_encoding:
            encoded_x = self.encode(x, apply_mask)
        else:
            encoded_x = x
        y = self.head(encoded_x)
#        if self.out_dim == 2 * self.patch_size:
        if self.multiplier != 1:
            bs, n_patch, _ = y.shape
            y = y.reshape(bs, n_patch, self.multiplier, self.patch_size).permute(0, 2, 1, 3)
        return y

    def encode(self, x: torch.Tensor, apply_mask: bool, return_patch_embedding=False, skip_patch_embedding=False) -> torch.Tensor:
        """
        Encodes the input signal into hidden space. Useful for downstream tasks
        :param x: input tensor
        :param apply_mask: if true, causal mask is applied
        :return: encoded input that is not proceed by prediction head
        """
        assert int(return_patch_embedding) + int(skip_patch_embedding) < 2, f'only one option allowed at max {return_patch_embedding=} {skip_patch_embedding=}'
        if not skip_patch_embedding:
            # 0. If the model is trained with Logit-Laplace loss, we first clamp the input
            if self.out_dim == 2 * self.patch_size:
                x = self.clamp(x)

            # 1. Patch embedding
            encoded_x = self.patch_embedding(x)  # |x| = (bs, n_patch, d_model)
            if return_patch_embedding:
                return encoded_x
        else:
            encoded_x = x

        # 2. Pass through GPT Layers
        for layer in self.gpt_layers:
            encoded_x = layer(encoded_x, apply_mask)
        return encoded_x

    def inference(self, ground_truth: torch.Tensor, context: int):
        assert len(ground_truth.shape) == 3, "Must reshape to (1, n_patches, patch_size)"
        assert context < ground_truth.shape[1]
        n_predictions = ground_truth.shape[1] - context
        x = ground_truth[:, :context, :]
        while n_predictions > 0:
            out = self.to_sequence(self.forward(x, apply_mask=False))
            # print(x.shape)
            # print(out.shape)
            # print(out[:, -1, :].shape)
            x = torch.concat((x, out[:, -1, :].unsqueeze(1)), dim=1)
            n_predictions -= 1
        x = x.reshape(-1)
        return x

    def to_sequence(self, out: torch.Tensor, keep_dim=False) -> torch.Tensor:
        """
        Maps model output to input space for visualization, only useful when using Logit-Laplace as loss function.
        :param out: model output
        :param keep_dim: If true, returns a tensor of shape |bs, 1, seq_len|, else |bs, seq_len|
        :return: Model prediction in input space
        """
        if keep_dim:
            return self.unclamp(self.sigmoid(out[:, 0, :]).unsqueeze(1))
        return self.unclamp(self.sigmoid(out[:, 0, :]))

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class GPT_with_linearOutput_2(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.pooling_fn = pooling_fn    # ['none-input-attn']
        self.use_attn_first_layer = use_attn_first_layer
        self.layer_output_index = layer_output_index
        self.use_all_layer_outputs = use_all_layer_outputs
        self.use_lora = use_lora
        self.fuse_feats = fuse_feats    # this just says if we are doing any fusion - must be true for any type of fusion
        self.use_conditional_lora = use_conditional_lora
        self.late_fusion = late_fusion
        self.turnoff_input_fuse = turnoff_input_fuse
        self.change_mha = change_mha

        print(f'{pretrained_path=}')
        if '19M' in pretrained_path and 'safetensors' not in pretrained_path:
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            if new_dropout_rate:
                state_dict['config']['dropout'] = new_dropout_rate
            self.gpt = GPT(**state_dict['config'])
    #        print(self.gpt)
            for key in list(state_dict['model']):
                if key.startswith('module.'):
                    state_dict['model']['.'.join(key.split('.')[1:])] = state_dict['model'].pop(key)
            self.gpt.load_state_dict(state_dict['model'])
    #        if new_dropout_rate:
    #            print('RUNNING set_dropout_rate()')
    #            set_dropout_rate(self.gpt, new_dropout_rate)
            d_model = state_dict['config']['d_model']
        elif '85M' in pretrained_path or '345M' in pretrained_path or ('19M' in pretrained_path and 'safetensors' in pretrained_path) or '1B' in pretrained_path or '10B' in pretrained_path:
            with open(model_config) as f:
                config = json.load(f)
            # init_dummy_ddp()  # Ensure this function is defined or remove if not needed
            if 'safetensors' in pretrained_path:
                state_dict = load_file_st(pretrained_path)  # Ensure load_file_st is defined
            elif '.pt' in pretrained_path:  # new format to store model
                state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    #                for key in list(state_dict['model_state_dict']):
    #                    if key.startswith('_orig_mod.'):
    #                        state_dict['model_state_dict']['.'.join(key.split('.')[1:])] = state_dict['model_state_dict'].pop(key)
                state_dict = state_dict['model_state_dict']
            else:   #
                state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
                state_dict = state_dict.module
                state_dict = state_dict.state_dict()
            if new_dropout_rate:
                config['dropout'] = new_dropout_rate
            self.gpt = GPT(**config)
            for key in list(state_dict):
                if key.startswith('_orig_mod.'):
                    state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
            if self.change_mha:
                state_dict = adjust_state_dict_for_per_head_model(state_dict, self.gpt)
            self.gpt.load_state_dict(state_dict)
            d_model = config['d_model']
        else:
            raise NotImplementedError(f'{pretrained_path=}')

        print(f'{count_parameters(self.gpt)=}')

        print('before lora', self.gpt)
#        freeze_nn(self.gpt)
#        freeze_nn(self.gpt.patch_embedding)
#        for ii in range(len(self.gpt.gpt_layers)//2): freeze_nn(self.gpt.gpt_layers[ii])

        if self.use_lora:
            peft_config = LoraConfig(r=4, lora_alpha=32, lora_dropout=0, target_modules=self.get_target_module_names(self.gpt) if self.change_mha else {'w_q', 'w_v'})     # scale in LoRA is defined as alpha/rank; dont tune alpha
            self.gpt = get_peft_model(self.gpt, peft_config)
            print('after lora', self.gpt)
            if self.use_conditional_lora:
                self.use_sequence = False
                self.use_learnable_component = True
                replace_lora_with_conditional_lora(self.gpt, layer_indices=None, lora_rank=4, lora_alpha=32, dropout=0, stft_dim=d_model, use_sequence=self.use_sequence, use_learnable_component=self.use_learnable_component, verbose=True)
#                # Conditional LoRA parameters
#                conditional_lora_kwargs = {
#                    'lora_rank': 4,
#                    'lora_alpha': 32,
#                    'lora_dropout': 0,
#                    'stft_dim': d_model,
#                    'use_sequence': False,
#                    'use_learnable_component': True
#                }
#                # Replace all layers and all heads
#                replace_lora_with_conditional_lora(
#                    model=self.gpt,
#                    conditional_lora_kwargs=conditional_lora_kwargs
#                )
                print('after conditional lora', self.gpt)

        if self.use_all_layer_outputs:
#            self.multiplier_1 = 4 # for 1:5, it is 4, default is 7
            self.multiplier_1 = len(self.gpt.gpt_layers)//3
        else:
            self.multiplier_1 = 1

#        self.norm1 = nn.BatchNorm1d(d_model)   # destroyed performance
        if self.late_fusion and self.fuse_feats:
            self.multiplier_2 = 1
        else:
            self.multiplier_2 = 1
            
        self.norm1 = nn.LayerNorm(d_model*self.multiplier_1)
        self.act1 = nn.ReLU()
        if 'input-attn' in pooling_fn:
            self.input_attn_linear = nn.Linear(d_model, 1)
        if pooling_fn == 'mean+std':
            self.linear = nn.Linear(d_model*2*self.multiplier_1*self.multiplier_2, out_classes)
        elif pooling_fn in ['mean', 'input-attn']:
            self.linear = nn.Linear(d_model*self.multiplier_1*self.multiplier_2, out_classes)
        elif 'none' in pooling_fn:
            self.linear = nn.Linear(31*d_model*self.multiplier_1*self.multiplier_2, out_classes, bias=True)    # seq dim 20, d_model=512
#            self.linear = nn.Linear(21*d_model*self.multiplier_1*self.multiplier_2, out_classes, bias=True)    # seq dim 20, d_model=512
#            self.linear = nn.Linear(3*d_model*self.multiplier_1*self.multiplier_2, out_classes, bias=True)    # seq dim 20, d_model=512
#            self.linear = nn.Linear(9*d_model*self.multiplier_1*self.multiplier_2, out_classes, bias=True)    # seq dim 20, d_model=512
#            self.linear = nn.Linear(11*d_model*self.multiplier_1*self.multiplier_2, out_classes, bias=True)    # seq dim 20, d_model=512
#            self.linear = nn.Linear(4*d_model*self.multiplier_1*self.multiplier_2, out_classes, bias=True)    # seq dim 20, d_model=512
#            self.linear = nn.Linear(49*d_model*self.multiplier_1*self.multiplier_2, out_classes)    # seq dim 20, d_model=512
#            self.linear = nn.Sequential(nn.Linear(21*d_model*self.multiplier_1*self.multiplier_2, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, out_classes))
#            self.linear = nn.Sequential(nn.Linear(31*d_model*self.multiplier_1*self.multiplier_2, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, out_classes))
##            self.linear = nn.Sequential(nn.Linear(31*d_model*self.multiplier_1*self.multiplier_2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, out_classes))
#            self.linear = nn.Linear(31*d_model*self.multiplier_1*self.multiplier_2, out_classes)  # much worse than below
#            self.linear = nn.Sequential(nn.Linear(31*d_model*self.multiplier_1*self.multiplier_2, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, out_classes))
        else:
            raise NotImplementedError(f'{self.pooling_fn=}')
        if self.use_attn_first_layer:
            self.attn_first_layer = MultiHeadAttention(512, 2)
        if self.fuse_feats:
#            self.fuse_feat_extractor = SpectralFeatureExtractor(40, d_model, use_e=True, normalize=True, fusion_method='gated')
            self.fuse_feat_extractor = SpectralPhaseFeatureExtractor(40, d_model, use_e=True, normalize=True, fusion_method='gated')    # this is what I call STFT features as it includes mag and phase
#            self.fuse_feat_extractor = WaveletFeatureExtractor(40, d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = MultiScaleSTFTPhaseFeatureExtractor(d_model=d_model, use_e=True, normalize=True, learnable_window=False, fusion_method='gated') # *
#            self.fuse_feat_extractor = MultiScaleSTFTPhaseFeatureExtractor(d_model=d_model, use_e=True, normalize=True, learnable_window=True, fusion_method='gated')
#            self.fuse_feat_extractor = MultiScaleOverlapSTFTPhaseFeatureExtractor(d_model=d_model, use_e=True, normalize=True, learnable_window=False, fusion_method='gated', patch_embedding=self.gpt.patch_embedding)
##            self.fuse_feat_extractor = pyPPGFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = pyPPGFeatureExtractorv2(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = PIIFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = ChronosFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = MomentFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = TTMFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = nkPPGFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
#            self.fuse_feat_extractor = MSTFTMomentFeatureExtractor(d_model=d_model, use_e=True, normalize=True, fusion_method='gated')
        else:
            self.fuse_feat_extractor = None
        if self.late_fusion:
            self.latefuser = AttentionFusionLayer()

    def get_target_module_names(self, model):
        target_modules = []
        n_layers = len(model.gpt_layers)
        n_heads = model.gpt_layers[0].attn.n_head  # Assuming all layers have the same number of heads
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                # For w_q
                target_modules.append(f"gpt_layers.{layer_idx}.attn.w_q_{head_idx}")
                # For w_v
                target_modules.append(f"gpt_layers.{layer_idx}.attn.w_v_{head_idx}")
        return target_modules

    def forward(self, x, get_gpt_output=False):   # B, N, P
        x_orig = x
        if type(x) in [list, tuple]:
            assert len(x) == 2
            aux_feats = x[1].detach().clone()
            x = x[0]
        else:
            aux_feats = None
#        print(x.shape)  # 64, 20 (num_patches), 40 (patchsize)
        if self.use_attn_first_layer:
            x = self.attn_first_layer(x, x, x)
#        if do_encode:
        # Compute spectral features and stft_emb before encoding
        e = None
        e = self.gpt.encode(x, apply_mask=False, return_all_encodings=False, fuse_feat_extractor=None, stft_emb=None)   # encoding of the time signal
        if self.fuse_feats and self.use_lora and self.use_conditional_lora:
            spectral_features = self.fuse_feat_extractor(x if aux_feats is None else aux_feats, e)  # Shape: (B, S+1, d_model)
            if self.use_sequence:
                stft_emb = spectral_features
            else:
                stft_emb = spectral_features.mean(dim=1)  # Shape: (B, d_model)
        else:
            stft_emb = None
        # Encode with stft_emb
        xo = x.detach().clone()
        x = self.gpt.encode(x, apply_mask=False, return_all_encodings=True, fuse_feat_extractor=self.fuse_feat_extractor if not self.turnoff_input_fuse else None, stft_emb=stft_emb, aux_feats=aux_feats)  # B, S, D
#        raise Exception(f'{[_.shape for _ in x]}=')
        if self.late_fusion:
            x = self.forward2(x, get_gpt_output=get_gpt_output)
            if get_gpt_output:
                x, x2 = x
            xx = self.gpt.encode(xo, apply_mask=False, return_all_encodings=True, fuse_feat_extractor=None, stft_emb=None)
            xx = self.forward2(xx, get_gpt_output=False)
#            x = torch.concatenate((x, xx), axis=-1)
            x = self.latefuser(xx, x)
            x = x.reshape(x.shape[0], -1)
#            x = x + xx
#            print(x.shape)
        else:
            x = self.forward2(x, get_gpt_output=get_gpt_output)
            if get_gpt_output:
                x, x2 = x

#        print(x_orig, x)
        x = self.linear(x)  # Shape: (B, out_classes)
#        print(x)
        if get_gpt_output:
            return (x, x2)
        else:
            return x


    def forward2(self, x, get_gpt_output=False):
        x0 = x[0]
        x3 = x[-1]
        if self.use_all_layer_outputs:
#            x = torch.concatenate(x[5:9], axis=-1)   # since embedding dimension is the last one
            x = torch.cat(x[len(self.gpt.gpt_layers)*1//3:len(self.gpt.gpt_layers)*2//3], dim=-1)   # Shape: (B, S', D*multiplier_1)
        else:
            x = x[self.layer_output_index]

        if get_gpt_output:
            x2 = self.gpt(x3, apply_mask=False, do_encode=False)

#        x = x.transpose(-1,-2)
        x = self.norm1(x)   # Shape: (B, S, D*multiplier_1)
#        x = x.transpose(-1,-2)
        x = self.act1(x)
        if 'input-attn' in self.pooling_fn:
            i_e = self.input_attn_linear(x0)
            i_e_sm = nn.functional.softmax(i_e, dim=1)
            x = x * i_e_sm
#            if self.multiplier_1 == 1:
#                x = x * i_e_sm
#            elif self.multiplier_1 > 1:
#                i_e_sm = i_e_sm.repeat(1,self.multiplier_1,1)
#                try:
#                    x = x * i_e_sm
#                except:
#                    raise Exception(f'{x.shape=} {i_e_sm.shape=}')
#            else: raise Exception
#            raise Exception(f'{x0.shape=} {i_e.shape=} {i_e_sm.shape=} {x.shape=} {i_e_sm[0].sum()=}') # x0.shape=torch.Size([1024, 21, 768]) i_e.shape=torch.Size([1024, 21, 1]) i_e_sm.shape=torch.Size([1024, 21, 1]) x.shape=torch.Size([1024, 21, 768])
        x = x[:,0:,:]   # Shape: (B, S, D*multiplier_1)
        if self.pooling_fn in ['mean', 'input-attn']:
            x = x.mean(1)  # Shape: (B, D*multiplier_1)
        elif self.pooling_fn == 'mean+std':
            x = torch.cat((x.mean(1), (x+1e-8).std(1)), dim=-1)  # Shape: (B, 2*D*multiplier_1)
        elif 'none' in self.pooling_fn:
#            raise Exception(f'{x.shape=}')  # for BS=64, seqlen=40, this is 64,21,512
#            x = x[:,1:,:]
            x = x.reshape(x.shape[0], -1)  # Shape: (B, S*D*multiplier_1)
        else:
            raise NotImplementedError(f'{self.pooling_fn=}')
        if get_gpt_output:
            return (x, x2)
        else:
            return x


class GPT(nn.Module):
    def __init__(self, patch_size: int, d_model: int, n_heads: int, n_layers: int,
                 dropout=0.1, max_len=2400, loss="mse", with_conv=False, out_dim_override=None, **kwargs):
        """
        GPT Model Definition
        """
        super().__init__()
#        assert loss == "mse" or "laplace"
        if kwargs:
            print(f"Ignored keyword arguments: {kwargs}")
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_head = n_heads
        self.n_layers = n_layers
        self.patch_embedding = PatchEmbedding(self.patch_size, self.d_model)
        self.gpt_layers = nn.ModuleList([
            GPTLayer(d_model=d_model, n_heads=n_heads, dropout=dropout, max_len=max_len, with_conv=with_conv)
            for _ in range(n_layers)
        ])
        self.out_dim = 2 * patch_size if "laplace" in loss else patch_size
        if out_dim_override:
            assert type(out_dim_override) is int
            assert 1 <= out_dim_override <= 10
            self.out_dim = out_dim_override * patch_size
        self.sigmoid = nn.Sigmoid()
        self.encoded_dim = d_model
        self.head = nn.Linear(self.d_model, self.out_dim)

    def forward(self, x: torch.Tensor, apply_mask: bool, do_encode=True) -> torch.Tensor:
        """
        Forward pass of GPT
        """
        if do_encode:
            encoded_x = self.encode(x, apply_mask)
        else:
            encoded_x = x
        try:
            y = self.head(encoded_x)
        except Exception as e:
            raise Exception(f'{e=} {self.head=} {encoded_x.shape=}')
        if self.out_dim == 2 * self.patch_size:
            bs, n_patch, _ = y.shape
            y = y.reshape(bs, n_patch, 2, self.patch_size).permute(0, 2, 1, 3)
        return y

    def encode(self, x: torch.Tensor, apply_mask: bool, return_all_encodings=False, fuse_feat_extractor=None, stft_emb: torch.Tensor = None, aux_feats=None, \
                ecg_input=None, ecg_patch_embedding=None, ppg_ecg_patch_emb_fuser=None, return_attn=False) -> torch.Tensor:
        """
        Encodes the input signal into hidden space with optional stft_emb
        """
#        lprint(f'\n c1: Encoding {x.shape=} {x.dim()=} {self.out_dim=} {self.patch_embedding=} {return_all_encodings=} \n')
        # If the model is trained with Logit-Laplace loss, clamp the input
#        lprint('x0')
        if self.out_dim == 2 * self.patch_size:
            if type(x) is torch.Tensor:
                x = self.clamp(x)
            else:
                assert type(x) in [tuple, list]
                x = list(x)
                for i in range(len(x)):
                    assert type(x[i]) is torch.Tensor
                    x[i] = self.clamp(x[i])

        if type(x) in [tuple, list]:
            dim_x = x[0].dim()
        else:
            dim_x = x.dim()

        # Patch embedding
        if dim_x == 3:
            if ecg_input:
                assert type(x) in [tuple, list]
                encoded_ppg = self.patch_embedding(x[0])
                encoded_ecg = self.patch_embedding(x[1]) #self.patch_embedding(x[1])  , ecg_patch_embedding(x[1])   # ecg_patch_embedding
                if not isinstance(ppg_ecg_patch_emb_fuser, GatedFusion):
                    encoded_x = torch.cat((encoded_ppg, encoded_ecg), dim=-1)
                    encoded_x = ppg_ecg_patch_emb_fuser(encoded_x)
                else:
                    encoded_x = ppg_ecg_patch_emb_fuser(encoded_ppg, encoded_ecg)
            else:
                encoded_x = self.patch_embedding(x)  # Shape: (B, S, D)
#            print(f'x1 {encoded_x.shape=}')
        elif dim_x == 4:
#            print('x2')
            B, C, S, P = x.shape
            x_reshaped = x.view(B * C, S, P)
            embedded = self.patch_embedding(x_reshaped)
            S = embedded.shape[1]
#            print(f'x21 {embedded.shape=}')
            embedded = embedded.view(B, C, S, -1)
#            print(f'x21 {embedded.shape=}')
            embedded = embedded.view(B, S, -1)
#            print(f'x21 {embedded.shape=}')
#            assert channel_mapper is not None
#            print(embedded.shape)
##            encoded_x = channel_mapper(embedded)
#            print(f'{embedded.shape=} {encoded_x.shape=}')
#            print('x22')
        else:
            raise NotImplementedError(f'{dim_x=}')

        if fuse_feat_extractor is not None:
            encoded_x = fuse_feat_extractor(x if aux_feats is None else aux_feats, encoded_x)

        if not return_all_encodings:
#            lprint(f'x22: {encoded_x.shape=}')
            return encoded_x
        all_encodings = [encoded_x]

        if return_attn:
            attns = []
        # Pass through GPT Layers with stft_emb
        for layer in self.gpt_layers:
            encoded_x = layer(encoded_x, apply_mask, stft_emb=stft_emb, return_attn=return_attn)
            if return_attn:
                assert type(encoded_x) is tuple and len(encoded_x) == 2, f'{type(encoded_x)=} {encoded_x.shape=}'
                encoded_x, attn = encoded_x
                attns.append(attn)
            all_encodings.append(encoded_x)

#        lprint('x1')
        if return_all_encodings:
#            lprint(f'{[_.shape for _ in all_encodings]=}')
            if return_attn:
                return (all_encodings, attns)
            else:
                return all_encodings
        else:
            if return_attn:
                return (encoded_x, attns)
            else:
#                lprint('x2')
                return encoded_x

#    def inference(self, ground_truth: torch.Tensor, context: int):
#        """
#        Inference method
#        """
#        assert len(ground_truth.shape) == 3, "Must reshape to (1, n_patches, patch_size)"
#        assert context < ground_truth.shape[1], "Context length exceeds sequence length."
#        n_predictions = ground_truth.shape[1] - context
#        x = ground_truth[:, :context, :]
#        while n_predictions > 0:
#            out = self.to_sequence(self.forward(x, apply_mask=False))
#            x = torch.cat((x, out[:, -1, :].unsqueeze(1)), dim=1)
#            n_predictions -= 1
#        x = x.reshape(-1)
#        return x
#
#    def to_sequence(self, out: torch.Tensor, keep_dim=False) -> torch.Tensor:
#        """
#        Maps model output to input space for visualization
#        """
#        if keep_dim:
#            return self.unclamp(self.sigmoid(out[:, 0, :]).unsqueeze(1))
#        return self.unclamp(self.sigmoid(out[:, 0, :]))

    @staticmethod
    def clamp(x, eps=0.1):
        return (1 - 2 * eps) * x + eps

    @staticmethod
    def unclamp(x, eps=0.1):
        return (x - eps) / (1 - 2 * eps)


class GPTLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout=0.1, max_len=2400, with_conv=False):
        """
        A single GPT Layer
        """
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(d_model, n_heads)  # Now expects stft_emb
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, drop_prob=dropout)
        self.mask = torch.tril(torch.ones(max_len, max_len))

        self.with_conv = with_conv
        self.conv = nn.Conv1d(d_model, d_model, 3, 1) if self.with_conv else None

    def forward(self, x: torch.Tensor, apply_mask: bool, stft_emb: torch.Tensor = None, return_attn=False) -> torch.Tensor:
        """
        Forward pass with optional stft_emb for Conditional LoRA
        """
        # Initialize mask
        bs = x.shape[0]  # batch size
        seq_len = x.shape[1]
        mask = self.mask[:seq_len, :seq_len].to(x.device)  # truncate mask to actual sequence length

        # Causal Convolution (if enabled)
        if self.with_conv:
            pad = torch.zeros((x.shape[0], 2, x.shape[2]), device=x.device)
            x = x + self.conv(torch.cat((pad, x), dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        # Attention Block (Pre-normalization)
        x_norm = self.norm_1(x)
        attn_output = self.attn(x_norm, x_norm, x_norm, mask=mask, return_attn=return_attn)
        if return_attn:
            assert type(attn_output) is tuple and len(attn_output) == 2
            attn_output, attn = attn_output
        x = x + self.dropout_1(attn_output)

        # Feed-Forward Network
        x_norm = self.norm_2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout_2(ffn_output)

        if return_attn:
            return (x, attn)
        else:
            return x


class RMSNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-8):
        """
        Initialize RMSNorm layer.
        :param num_features: Number of features in the input.
        :param epsilon: Small value to prevent division by zero.
        """
        super(RMSNorm, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        """
        Forward pass of the RMSNorm.
        :param x: Input tensor.
        :return: Normalized tensor.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        x = x / rms
        return self.scale * x
