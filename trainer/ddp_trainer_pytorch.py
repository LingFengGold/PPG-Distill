# IMPLEMENTS PYTORCH DDP TRAINING
import os
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
# import gpustat
import logging
logging.basicConfig(level=logging.INFO)
from logging import info as lprint
import math

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from data.pretrain_dataset import PretrainDataset
from trainer.utils import *
from model.gpt import GPT
from contextlib import contextmanager
from torch.nn.parallel import DistributedDataParallel as ddp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc, matthews_corrcoef, roc_auc_score, average_precision_score


def rankme_loss_gpu(Z, epsilon=1e-7):
    """
    Computes RankMe for batch embeddings (B, K) on GPU.
    
    Args:
        Z (torch.Tensor): Embedding matrix of shape (B, K)
        epsilon (float): Numerical stability term
    
    Returns:
        float: RankMe score for the batch
    """
    # Compute singular values (shape: [min(B,K)])
    sigma = torch.linalg.svdvals(Z)  # GPU-optimized
    
    # Normalize and compute entropy
    sigma_norm = sigma / (sigma.sum() + epsilon)
    entropy = -torch.sum(sigma_norm * torch.log(sigma_norm + epsilon))
    
    return -torch.exp(entropy).mean()


class GPT_Trainer:
    def __init__(self, model: GPT, optim: torch.optim.Optimizer, train_loader: DataLoader, save_path: str, use_ddp: bool,
                 config_file_path: str or dict, log_freq=100, scheduler=None, loss_fn=None, loss_log=None, AMP=True, ft=False, train_config=None, val_loader=None, task_type=None):
        # initialization
#        assert task_type in ['regression', 'classification']
        assert loss_fn in ["mse", "laplace", 'ce', 'ce+laplace', 'mse+laplace']
        self.snapshot_path = save_path+"_snapshot.pth"
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if use_ddp else 0
        self.device = f"cuda:{self.gpu_id}"
        self.model = model.to(self.device)
        self.data_loader = train_loader
        self.optim = optim
        self.save_path = save_path
        self.log_freq = log_freq
        self.ft = False if str(ft).lower() == 'false' else True
        self.loss_fn_name = loss_fn
        print(f'{self.loss_fn_name=}')
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'laplace':
            self.loss_fn = LogitLaplaceLoss()
        elif loss_fn == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'ce+laplace':
#            self.loss_fn = Added_Loss_Functions([nn.CrossEntropyLoss(), LogitLaplaceLoss()])
#            self.loss_fn.weights = [1, 1]   # tried several combinations but just adding is the most robust to diff. experiment choices like BS, LR, etc.; BUT for freezed GPT model, higher value of laplace is needed
            self.loss_fn = [nn.CrossEntropyLoss(label_smoothing=0.0), LogitLaplaceLoss()]   # for MAUS, label_smoothing=0.1 is good
        elif loss_fn == 'mse+laplace':
            self.loss_fn = [nn.L1Loss(), LogitLaplaceLoss()]    # [nn.L1Loss(), LogitLaplaceLoss()]    nn.MSELoss; huber was bad
        else:
            raise NotImplementedError(f'{loss_fn=}')
        self.scheduler = scheduler
        self.epoch = 0
        self.use_ddp = use_ddp
        if os.path.exists(self.snapshot_path):
            print("Loading Previous Checkpoint")
            self._load_snapshot(self.snapshot_path)
        self.loss_log = [] if not loss_log else loss_log
        self.AMP = AMP
        self.scaler = torch.amp.GradScaler('cuda')
        if isinstance(config_file_path, str):
            with open(config_file_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = config_file_path
        self.model = ddp(model, [self.gpu_id], find_unused_parameters=True) if use_ddp else model
        self.train_config = train_config
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        print("Total Trainable Parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        ##### NEW VALIDATION-RELATED PARAMS #####
        self.val_loader = val_loader
        self.task_type = task_type
        self.best_val_metric = float('inf') if task_type == 'regression' else -np.inf
        self.best_epoch = -1
        #########################################
        self.aux_w = nn.Parameter(torch.tensor(torch.randn(1)).to(self.device))
        self.optim.add_param_group({"params": [self.aux_w]})  # ✅ Proper format

    def train(self, num_epochs: int, save_freq=5): #visualization_dataset=None, save=True, display_freq=1, save_freq=5):
        for i in range(self.epoch, num_epochs):
            epoch=i
            self.fit(i, num_epochs)

            ##### VALIDATION LOGIC #####
            if (epoch + 1) % 10 == 0 and self.val_loader is not None:
                val_metric = self.validate()
                is_better = (self.task_type == 'regression' and val_metric < self.best_val_metric) or \
                            (self.task_type == 'classification' and val_metric > self.best_val_metric)
                
                if is_better:
                    self.best_val_metric = val_metric
                    self.best_epoch = epoch
                    self._save_best_model()
                    lprint(f"New best model at epoch {epoch+1} with val metric: {val_metric:.4f}")
            ###########################

            if ((i+1) % save_freq == 0 and self.gpu_id == 0 and i != 0) or i == (num_epochs-1):
                self._save_snapshot() # if save else self.unsaved_warning()

    def fit(self, epoch, num_epochs):
        self.model.train()
        total = len(self.data_loader)
        data_iter = tqdm.tqdm(enumerate(self.data_loader),
                              desc="EP:%d" % epoch,
                              total=total,
                              bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0

        accumulation_steps = 1  # Number of steps to accumulate gradients

        for i, data in data_iter:
            # send everything to cuda and get model output
            data = {key: value.to(self.device) for key, value in data.items()}

            # Zero gradients only at the start of accumulation
            if i % accumulation_steps == 0:
                self.optim.zero_grad()

            with self.optional_autocast():
                # get model output
                if self.ft:
                    if 'ce+' in self.loss_fn_name:
#                        print(data["ppg_segments"].shape)
                        out1, out2 = self.model(data["ppg_segments"], get_gpt_output=True)
                        label1 = data['ft_label'].long()
                        out2, label2 = self.process(out2, data)
                    elif 'mse+' in self.loss_fn_name:
                        if self.train_config and 'ecg_input' in self.train_config and self.train_config['ecg_input']:
                            input_ = (data["ppg_segments"], data["ecg_segments"])
                        elif self.train_config and 'feats_train' in self.train_config:
                            input_ = (data["ppg_segments"], data["aux_feats"])
                        else:
                            input_ = data["ppg_segments"]
                        out1, out2 = self.model(input_, get_gpt_output=True)
                        try:
                            label1 = data['ft_label']
                        except Exception as e:
                            raise Exception(f'{e=} {data=}')
#                        print(label1)
                        out2, label2 = self.process(out2, data)
                    else:
                        out = self.model(data["ppg_segments"])
                        label = data['ft_label'].long()
                else:
                    out = self.model.forward(data["ppg_segments"], apply_mask=True)
                    # preprocess
                    out, label = self.process(out, data)
                
                # calculate loss
                if self.loss_fn_name in ['ce+laplace', 'mse+laplace']:
                    main_loss = self.loss_fn[0](out1.squeeze(1), label1)
                    aux_loss = self.loss_fn[1](out2, label2)
##                    raise Exception(f'{data["ppg_segments"]=} {torch.isnan(data["ppg_segments"]).sum()=} {torch.isposinf(data["ppg_segments"]).sum()=} {torch.isneginf(data["ppg_segments"]).sum()=} {out1=} {out2=} {label1=} {label2=} {main_loss=} {aux_loss=}')
##                    lprint(f'{out2.shape=}')    # out2.shape=torch.Size([32, 2, 1200])
#                    out2_flattened = out2.view(out2.shape[0], -1)
#                    rankme_loss = rankme_loss_gpu(out2_flattened)
#                    loss = main_loss + aux_loss    # BAD
#                    loss = main_loss + self.aux_w * aux_loss # + 0.001*(self.aux_w**2)     # BAD
#                    loss = main_loss + (1 - (epoch + 1) / num_epochs) * self.aux_w.exp() * aux_loss + 0.001*(self.aux_w**2)# 10 * (1 - (epoch + 1) / num_epochs) * aux_loss # + rankme_loss
#                    loss = main_loss + self.aux_w.exp() * aux_loss + 0.01*(self.aux_w**2)# 10 * (1 - (epoch + 1) / num_epochs) * aux_loss # + rankme_loss
#                    loss = main_loss + torch.sigmoid(self.aux_w) * aux_loss + 0.01*(self.aux_w**2)# 10 * (1 - (epoch + 1) / num_epochs) * aux_loss # + rankme_loss
#                    loss = main_loss + torch.log(1 + torch.exp(self.aux_w)) * aux_loss + 0.01*(self.aux_w**2)# 10 * (1 - (epoch + 1) / num_epochs) * aux_loss # + rankme_loss
#                    loss = main_loss + aux_loss # + rankme_loss    # DEBATABLE BUT OK SOMETIMES
#                    loss = main_loss + 1 * (1 - (epoch + 1) / num_epochs) * aux_loss # + rankme_loss   # DEFAULT - ok performance
                    loss = main_loss + 10 * (1 - (epoch + 1) / num_epochs) * aux_loss # + rankme_loss  # SOMETIMES IS BEST
                else:
                    loss = self.loss_fn(out, label)

            # Accumulate gradients
            if self.AMP:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step the optimizer every `accumulation_steps`
            if (i + 1) % accumulation_steps == 0:
                if self.AMP:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()
                self.scheduler.step()

                # Zero gradients for the next accumulation step
                self.optim.zero_grad()

            avg_loss += loss.item()

        # After the loop, if there are remaining gradients, step the optimizer
        if (i + 1) % accumulation_steps != 0:
            if self.AMP:
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.step()
            self.scheduler.step()

            # write postfix
            post_fix = {"epoch": epoch, "iter": i, "avg_loss": avg_loss / (i + 1)}
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
            if i % 20 == 0:
                self.loss_log.append(loss.item())

        print("EP%d, avg_loss=%f" % (epoch, avg_loss / len(data_iter)))
        #print(gpustat.main())        
#        lprint(gpustat.GPUStatCollection.new_query())

        self.epoch += 1
        return avg_loss

    ##### NEW VALIDATION METHOD #####
    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm.tqdm(self.val_loader, desc="Validating"):
                data = {key: value.to(self.device) for key, value in data.items()}
                
                # Forward pass matching training logic
                if self.ft:
                    if 'ecg_input' in self.train_config and self.train_config['ecg_input']:
                        input_ = (data["ppg_segments"], data["ecg_segments"])
                    else:
                        input_ = data["ppg_segments"]
                        
                    if '+laplace' in self.loss_fn_name:
                        out1, _ = self.model(input_, get_gpt_output=True)  # Get main output only
                    else:
                        out1 = self.model(input_)
                else:
                    out = self.model.forward(data["ppg_segments"], apply_mask=False)
                    out, label = self.process(out, data)
#                print('x11') 
                # Handle different task types
                if self.task_type == 'regression':
                    all_preds.append(out1.cpu().numpy())
                    all_labels.append(data['ft_label'].cpu().numpy())
                else:  # Classification
                    preds = torch.argmax(out1, dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(data['ft_label'].cpu().numpy())

        # Calculate validation metric
        try:
            all_preds = np.concatenate(all_preds)
        except Exception as e:
            raise Exception(f'{e=} {self.val_loader=}')
        all_labels = np.concatenate(all_labels)
        
        if self.task_type == 'regression':
            return mean_squared_error(all_labels, all_preds)
        else:
            return accuracy_score(all_labels, all_preds)
    ##################################

    ##### NEW BEST MODEL SAVING #####
    def _save_best_model(self):
        best_path = self.save_path + "_best.pth"
        model_cp = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
        torch.save({
            'model': model_cp,
            'epoch': self.best_epoch,
            'val_metric': self.best_val_metric
        }, best_path)
    ##################################

    def process(self, out, data):
        if out.shape[1] != 2:  # in this case MSE loss is used
            label = data["label"]
            out = out[:, :-1, :]  # we drop the last token prediction
        else:  # in this case Logit-Laplace loss is used
            label = data["label"].reshape(data["label"].shape[0], 1, -1)
            if '+' in self.loss_fn_name: # in ['ce+laplace', 'mse+laplace']:
                label = self.model.module.gpt(label) if self.use_ddp else self.model.gpt.clamp(label)
            else:
                label = self.model.module.clamp(label) if self.use_ddp else self.model.clamp(label)
            try:
                out = out[:, :, :-1, :]  # we drop the last token prediction
            except Exception as e:
                raise Exception(f'{e=} {out.shape=}')   # out.shape=torch.Size([32, 2, 40])
            out = out.reshape(out.shape[0], 2, -1)
        return out, label

    def _load_snapshot(self, checkpoint_path):
        cp = torch.load(checkpoint_path)
        self.model.load_state_dict(cp["model"])
        self.epoch = cp["epoch"]
        self.scheduler.load_state_dict(cp["lr_schedule"])
        self.optim.load_state_dict(cp["optimizer"])

    def _save_snapshot(self):
        """
        Saves the model checkpoint
        """
        path = self.save_path + f"_{self.epoch-1}.pth"
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_cp = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
        snapshot = {
            'config': self.config,
            'epoch': self.epoch,
            'model': model_cp,
            'optimizer': self.optim.state_dict(),
            'lr_schedule': self.scheduler.state_dict(),
            'loss_log': self.loss_log
        }
        torch.save(snapshot, path)
#        torch.save(snapshot, self.snapshot_path)
        print(f"Model checkpoint created at ep {self.epoch-1} on {path}")

    def save_log(self, path: str):
        torch.save(self.loss_log, path)

    def plot_loss(self):
        plt.plot(self.loss_log)
        plt.show()

    @staticmethod
    def unsaved_warning():
        print("Model Not Saved.")

    @contextmanager
    def optional_autocast(self):
        if self.AMP:
            with torch.amp.autocast('cuda'):
                yield
        else:
            yield
