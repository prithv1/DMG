import os
import sys
import json
import copy
import time
import torch
import random
import argparse
import torchvision
import dataloaders
import torch.utils.data

import numpy as np

from tqdm import tqdm
from config import Config
from torch import nn, optim
from pprint import pprint, pformat
from torch.autograd import Variable
from torch.nn import functional as F
from utils.inverse_lr_scheduler import InvLR
from torchvision import datasets, transforms

# Import all the model definitions
from models import Basic_Model, MultiHead_Model, SubNetwork_SuperMask_Model, SuperMask

# Import Dataloaders
from dataloaders import DomainDataset, Aggregate_DomainDataset

# Import all the trainers and evaluators
from trainers import Aggregate_Trainer, MultiHead_Trainer, SubNetwork_SuperMask_Trainer

parser = argparse.ArgumentParser("Run training for a particular phase")

parser.add_argument(
    "--phase",
    required=True,
    choices=["aggregate_training", "multihead_training", "supermask_training"],
    help="Which phase to train, this must match the 'PHASE' parameter in the provided config.",
)

parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for a training job"
)

parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like nesting) using a dot operator. The actual config will be updated and recorded in the serialization directory.",
)

if __name__ == "__main__":
    # Parse arguments
    _A = parser.parse_args()

    # Create a config with default values, then override from config file and _A.
    # This config object is immutable, nothing can be changed in this anymore
    _C = Config(_A.config_yml, _A.config_override)

    # Match the phase from arguments and config parameters
    if _A.phase != _C.HJOB.PHASE:
        raise ValueError(
            f"Provided `--phase` as {_A.phase}, does not match config PHASE ({_C.HJOB.PHASE})."
        )

    # Print configs and args
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Display config to be used
    print(_C)

    # Get environment name
    ENV_NAME = _C.get_env()
    pprint(ENV_NAME)

    # Get checkpoint directory
    CKPT_DIR = _C.CKPT.STORAGE_DIR + _C.DATA.DATASET + "/checkpoints/" + ENV_NAME

    # Create directory and save config
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    _C.dump(os.path.join(CKPT_DIR, "config.yml"))

    # Fix seeds for reproducibility
    # Reference - https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.HJOB.RANDOM_SEED)
    torch.manual_seed(_C.HJOB.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.HJOB.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Model Definition
    if _A.phase == "aggregate_training":
        model = Basic_Model.from_config(_C)
        input_size = model.input_size
    elif _A.phase == "multihead_training":
        model = MultiHead_Model.from_config(_C)
        input_size = model.input_size
    elif _A.phase == "supermask_training":
        MASK_LAYERS = [_C.MODEL.MASK_LAYERS]
        if "," in _C.MODEL.MASK_LAYERS:
            MASK_LAYERS = _C.MODEL.MASK_LAYERS.split(",")
        MASK_LAYERS = sorted(MASK_LAYERS)
        # Define joint model
        joint_model = Basic_Model.from_config(_C)
        input_size = joint_model.input_size
        # Define conditional computation model
        model = SubNetwork_SuperMask_Model.from_config(_C, joint_model, MASK_LAYERS)
        # Get init arguments for the policy modules
        act_size = model.get_mask_struct(_C.MODEL.POLICY_CONV_MODE)
        # Create policy modules
        policy_modules = []
        for i in range(len(MASK_LAYERS)):
            policy_modules.append(SuperMask.from_config(_C, act_size[i]))
    else:
        print("Training phase not supported")

    # Move model to GPU
    if _C.PROCESS.USE_GPU:
        model.cuda()
        if _A.phase == "supermask_training":
            for i in range(len(MASK_LAYERS)):
                policy_modules[i].cuda()

    # Data Transformations
    data_transforms = {
        "val": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    if _C.DATA.DATASET == "PACS":
        data_transforms["train"] = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif _C.DATA.DATASET == "DomainNet":
        data_transforms["train"] = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    DOMAIN_LIST = _C.DATA.DOMAIN_LIST
    if "," in _C.DATA.DOMAIN_LIST:
        DOMAIN_LIST = _C.DATA.DOMAIN_LIST.split(",")
    else:
        DOMAIN_LIST = [_C.DATA.DOMAIN_LIST]

    TARGET_DOMAINS = _C.DATA.TARGET_DOMAINS
    if "," in _C.DATA.TARGET_DOMAINS:
        TARGET_DOMAINS = _C.DATA.TARGET_DOMAINS.split(",")
    else:
        TARGET_DOMAINS = [_C.DATA.TARGET_DOMAINS]

    # Dataloader Definitions
    train_split_obj = Aggregate_DomainDataset(
        _C.DATA.DATASET,
        DOMAIN_LIST,
        _C.DATA.DATA_SPLIT_DIR,
        "train",
        data_transforms["train"],
        _C.DATALOADER.BATCH_SIZE,
        _C.PROCESS.NUM_WORKERS,
        _C.PROCESS.USE_GPU,
        shuffle=True,
    )

    val_split_obj = Aggregate_DomainDataset(
        _C.DATA.DATASET,
        DOMAIN_LIST,
        _C.DATA.DATA_SPLIT_DIR,
        "val",
        data_transforms["val"],
        _C.DATALOADER.BATCH_SIZE,
        _C.PROCESS.NUM_WORKERS,
        _C.PROCESS.USE_GPU,
        shuffle=False,
    )

    test_split_obj = Aggregate_DomainDataset(
        _C.DATA.DATASET,
        TARGET_DOMAINS,
        _C.DATA.DATA_SPLIT_DIR,
        "test",
        data_transforms["test"],
        _C.DATALOADER.BATCH_SIZE,
        _C.PROCESS.NUM_WORKERS,
        _C.PROCESS.USE_GPU,
        shuffle=False,
    )

    # Setup optimizers and start training
    if _A.phase == "aggregate_training" or _A.phase == "multihead_training":
        parameters = model.parameters()
        if _C.OPTIM.OPTIMIZER == "Adam":
            optimizer = optim.Adam(
                parameters,
                lr=_C.OPTIM.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=_C.OPTIM.WEIGHT_DECAY,
            )
        else:
            print("Optimizer not supported yet")

        if _C.OPTIM.LEARNING_RATE_SCHEDULER == "exp":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, _C.OPTIM.LEARNING_RATE_DECAY_RATE
            )
        elif _C.OPTIM.LEARNING_RATE_SCHEDULER == "invlr":
            scheduler = InvLR(optimizer)
        else:
            print("LR Scheduler not identified")

    elif _A.phase == "supermask_training":
        model_parameters = model.parameters()
        policy_module_parameters = []
        for module in policy_modules:
            policy_module_parameters += list(module.parameters())

        if _C.OPTIM.OPTIMIZER == "Adam":
            model_optimizer = optim.Adam(
                model_parameters,
                lr=_C.OPTIM.MODEL_LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=_C.OPTIM.MODEL_WEIGHT_DECAY,
            )

            policy_optimizer = optim.Adam(
                policy_module_parameters,
                lr=_C.OPTIM.POLICY_LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=_C.OPTIM.POLICY_WEIGHT_DECAY,
            )
        else:
            print("Optimizer not supported yet")

        if _C.OPTIM.LEARNING_RATE_SCHEDULER == "exp":
            model_scheduler = optim.lr_scheduler.ExponentialLR(
                model_optimizer, _C.OPTIM.LEARNING_RATE_DECAY_RATE
            )
            policy_scheduler = optim.lr_scheduler.ExponentialLR(
                policy_optimizer, _C.OPTIM.LEARNING_RATE_DECAY_RATE
            )
        elif _C.OPTIM.LEARNING_RATE_SCHEDULER == "invlr":
            model_scheduler = InvLR(model_optimizer)
            policy_scheduler = InvLR(policy_optimizer)
        else:
            print("LR Scheduler not identified")
    else:
        print("Phase not supported yet")

    # Define trainer objects and start training
    if _A.phase == "aggregate_training":
        Trainer = Aggregate_Trainer(
            model,
            _C.PROCESS.USE_GPU,
            train_split_obj,
            val_split_obj,
            test_split_obj,
            _C.EP_IT.MAX_EPOCHS,
            optimizer,
            ENV_NAME,
            _C.EP_IT.LOG_INTERVAL,
            scheduler,
            _C.OPTIM.LEARNING_RATE_DECAY_MODE,
            _C.OPTIM.LEARNING_RATE_DECAY_STEP,
            CKPT_DIR,
            _C,
        )

        Trainer.train(DOMAIN_LIST, _C.EP_IT.CKPT_STORE_INTERVAL)

    elif _A.phase == "multihead_training":
        Trainer = MultiHead_Trainer(
            model,
            _C.PROCESS.USE_GPU,
            train_split_obj,
            val_split_obj,
            test_split_obj,
            _C.MODEL.TRAIN_FORWARD_MODE,
            _C.MODEL.EVAL_FORWARD_MODE,
            _C.EP_IT.MAX_EPOCHS,
            optimizer,
            ENV_NAME,
            _C.EP_IT.LOG_INTERVAL,
            scheduler,
            _C.OPTIM.LEARNING_RATE_DECAY_MODE,
            _C.OPTIM.LEARNING_RATE_DECAY_STEP,
            CKPT_DIR,
            _C,
        )

        Trainer.train(DOMAIN_LIST, _C.EP_IT.CKPT_STORE_INTERVAL)
    elif _A.phase == "supermask_training":

        Trainer = SubNetwork_SuperMask_Trainer(
            model,
            MASK_LAYERS,
            policy_modules,
            DOMAIN_LIST,
            TARGET_DOMAINS,
            train_split_obj,
            val_split_obj,
            test_split_obj,
            _C.EP_IT.MAX_EPOCHS,
            model_optimizer,
            policy_optimizer,
            CKPT_DIR,
            _C,
            model_scheduler,
            policy_scheduler,
            _C.OPTIM.LEARNING_RATE_DECAY_MODE,
            _C.EP_IT.LOG_INTERVAL,
            ENV_NAME,
            _C.PROCESS.USE_GPU,
        )

        Trainer.train(_C.EP_IT.CKPT_STORE_INTERVAL)
    else:
        print("Training phase not identified.")
