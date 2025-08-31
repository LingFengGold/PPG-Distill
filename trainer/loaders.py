import numpy as np
import sys
sys.path.append('.')
sys.path.append('./MARS/MARS/optimizers')
# from mars import MARS
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from trainer.ddp_trainer_pytorch import *
from model.gpt import *


def load_trainer(model, optim, train_data_loader, save_path, train_config,
                 lr_scheduler, model_config, loss_log, val_loader=None, task_type=None):
    trainer = GPT_Trainer(model, optim, train_data_loader, save_path, train_config["DDP"], model_config,
                          train_config["log_freq"], lr_scheduler, model_config["loss"], loss_log,
                          train_config["AMP"], train_config["ft"], train_config=train_config, val_loader=val_loader, task_type=task_type)
    return trainer


def load_train_objs(model, train_config, model_config, train_dataset_path, \
                    train_labels_dataset_path='', train_sampler=None, data_red_factor=1, train_data_path_ecg=None, \
                    val_dataset_path=None, val_labels_dataset_path=None, val_data_path_ecg=None):
    print("Loading Train Dataset", train_dataset_path)
    train_dataset = PretrainDataset(train_dataset_path, model_config["patch_size"], instance_normalization=False,
                                    train_labels_dataset_path=train_labels_dataset_path, train_config=train_config, data_red_factor=data_red_factor, train_data_path_ecg=train_data_path_ecg)

    print("Creating Dataloader")
    train_dataloader = DataLoader(train_dataset, train_config["batch_size"], shuffle=False, sampler=train_sampler, drop_last=True)

    #
    val_dataloader = None
    if val_dataset_path or val_labels_dataset_path:
        assert int(val_dataset_path is None) + int(val_labels_dataset_path is None) < 2, f'one of them is missing!: {val_dataset_path=} {val_labels_dataset_path=}'
        val_dataset = PretrainDataset(val_dataset_path, model_config["patch_size"], instance_normalization=False,
                                        train_labels_dataset_path=val_labels_dataset_path, train_config=train_config, data_red_factor=data_red_factor, train_data_path_ecg=val_data_path_ecg)
        val_dataloader = DataLoader(val_dataset, train_config["batch_size"] // 4, shuffle=False, sampler=None, drop_last=True)  # //4 because val set is small sometimes

    print("Setting Up Optimizer & LR Schedule")
    # Reading LR parameters
    lr_max, lr_init, lr_final, lr_schedule_ratio, lr_warm_up = (
        train_config["lr_max"], train_config["lr_init"], train_config["lr_final"],
        train_config["lr_schedule_ratio"], train_config["lr_warm_up"]
    )
    lr_schedule_step = int(lr_schedule_ratio * train_config["epochs"] * len(train_dataloader))

    # Reading optimizer parameters
    adam_beta1, adam_beta2, adam_weight_decay = (
        train_config["adam_beta1"], train_config["adam_beta2"], train_config["adam_weight_decay"]
    )

    # Building optimizer
    optim = Adam(model.parameters(), lr=lr_init, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay)

    # Building LR scheduler
    warm_up_steps = int(lr_schedule_step * lr_warm_up)
    lambda_warmup = lambda step: (lr_init + step * (lr_max - lr_init) / warm_up_steps) / lr_init
    warm_up_scheduler = LambdaLR(optim, lr_lambda=lambda_warmup)
    main_scheduler = CosineAnnealingLR(optim, T_max=lr_schedule_step - warm_up_steps, eta_min=lr_final)
    lr_scheduler = LrScheduler(optim, warm_up_scheduler, main_scheduler, warm_up_steps, lr_schedule_step)

    return model, train_dataloader, val_dataloader, model_config, train_config, optim, lr_scheduler


def load_config_file(config_path: str):
    # open the json file
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_untrained_model(model_config_path: str, initialization=None):
    # open the json file
    with open(model_config_path, "r") as f:
        config = json.load(f)

    # create model instance
    if 'name' in config:
        name = config['name']
        if config['name'] == 'GPT_with_linearOutput':
            del config['name']
            model = GPT_with_linearOutput(pretrained_path=config['pretrained_path'], model_config=config['model_config'], out_classes=config['out_classes']) #**config)
        elif config['name'] == 'GPT_fuse':
            del config['name']
            model = GPT_fuse(pretrained_path=config['pretrained_path'], model_config=config['model_config'], out_classes=config['out_classes']) #**config)
        else:
            raise NotImplementedError(f'{name=}')
    else:
        model = GPT(**config)
        # initialize the model
        if initialization == "xavier":
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    return model


def load_trained_model(model_checkpoint_path: str):
    # load .pth file
    cp = torch.load(model_checkpoint_path)

    # load model config file from checkpoint
    config = cp["config"]

    # create model instance
    model = GPT(**config)

    # load trained weights
    model.load_state_dict(cp["model"])
    return model


def load_pretrained_checkpoint(model_checkpoint_path: str):
    cp = torch.load(model_checkpoint_path)
    return cp
