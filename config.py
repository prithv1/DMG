"""
A module for package-wide configuration
management. Inspired by Ross Girchick's yacs template
Also, 
kd's source -- https://github.com/kdexd/probnmn-clevr/blob/master/probnmn/config.py
"""
from typing import List, Any
from yacs.config import CfgNode as CN


class Config(object):
    """
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be over-written by (first) through a YAML file and (second) through
    a list of attributes and values.

    - This class definition contains details relevant to all the training phases
        but is listed as the final adaptation training phase
    
    Parameters
    ===========
    config_yaml: str
        Path to a YAML file containing configuration parameters to override
    config_override: List[Any], optional (default=[])
        A list of sequential attributes and values of parameters to override. This happens
        after overriding from YAML file.

    Attributes
    ===========
    ---------------------------
    (HIGH-LEVEL JOB RELATED ARGUMENTS)

    HJOB.RANDOM_SEED: 123
        Random seed for numpy and PyTorch for reproducibility
    
    HJOB.PHASE: "aggregate_training"
        Which phase to train on? One of
        - ``aggregate_training''
        - ``multihead_training''
        - ``supermask_training''

    HJOB.JOB_STRING: "test_job"
        Job string prefix

    HJOB.WANDB_PROJECT: "DMG"
        Project name from wandb

    HJOB.WANDB_DIR: "wandb_runs/"
        Directory to store wandb data in

    --------------------------
    (DATA RELATED ARGUMENTS)
    
    DATA.DATASET: "PACS"
        Dataset to perform experiments on

    DATA.DOMAIN_LIST: "cartoon,photo,sketch"
        List of domains to train jointly on

    DATA.TARGET_DOMAINS: "art_painting"
        List of target domains to evaluate

    DATA.DATA_SPLIT_DIR: "data/"
        Directory which stores all the data

    DATA.HEAD_MODE: "single"
        Specific mode of the aggregate dataloaders, specific to multi-head models.
        Can be either "single" or "multi".

    -------------------------
    (CHECKPOINT RELATED ARGUMENTS)

    CKPT.STORAGE_DIR: "../DMG/"
        Directory to store checkpoints in

    -------------------------
    (MODEL DEFINITION RELATED ARGUMENTS)

    MODEL.BASE_MODEL: "alexnet"
        The base architecture on which proposed model definition is to be built upon

    MODEL.PARAM_INIT: "custom"
        Whether to initialize new parameters in a standard-vs-custom matter
    
    MODEL.USE_PRETRAINED: True
        Whether to use pre-trained base model

    MODEL.SPLIT_LAYER: "classifier.0"
        The layer at which one should split old/new parameters for finetuning

    MODEL.TRAIN_FORWARD_MODE: "route"
        How to forward pass instances for the multi-headed network during training

    MODEL.EVAL_FORWARD_MODE: "route"
        How to forward pass instances for the multi-headed network during evaluation

    MODEL.NUM_CLASSES: 7
        Number of output classes for the classification task

    MODEL.MASK_LAYERS: "classifier.6"
        Comma-separated names of layers at which the conditional computation mask is to be applied

    MODEL.POLICY_SAMPLE_MODE: "sample"
        Sampling mode of the layer-wise mask policies -- ["sample", "greedy"]

    MODEL.POLICY_CONV_MODE: False
        Set to True, when a shared mask per unit in a channel is applied

    MODEL.MASK_INIT_SETTING: random
        How to initialize the masks -- ["random_uniform", "scalar"]

    MODEL.MASK_INIT_SCALAR: 1.0
        Scalar to initialize the masks with -- 1.0 (by default)

    -------------------------
    (DATALOADER RELATED ARGUMENTS)

    DATALOADER.BATCH_SIZE: 64
        Batch size for the dataloader

    DATALOADER.DATA_SAMPLING_MODE: "uniform"
        Whether to sample data in a uniform / balanced manner

    -------------------------
    (OPTIMIZATION RELATED ARGUMENTS)

    OPTIM.OPTIMIZER: Adam
        Optimizer to use -- [Adam, SGD]

    OPTIM.LEARNING_RATE: 5e-4
        Learning rate to use

    OPTIM.LEARNING_RATE_DECAY_RATE: 0.96
        Decay rate to use for learning rate decay

    OPTIM.LEARNING_RATE_DECAY_MODE: "iteration"
        Whether to decay learning rate per-iteration ("iteration") or per-epoch ("epoch")

    OPTIM.LEARNING_RATE_DECAY_STEP: 15000
        If we're decaying learning rate per-iteration, what is the decay-step size?

    OPTIM.LEARNING_RATE_SCHEDULER: exp
        What kind of learning rate scheduler to use

    OPTIM.WEIGHT_DECAY: 1e-5
        Weight decay to use

    OPTIM.MODEL_LEARNING_RATE: 5e-4
        Learning rate to use for the base model during meta-train updates

    OPTIM.POLICY_LEARNING_RATE: 5e-4
        Learning rate to use for the mask-policies

    OPTIM.POLICY_WEIGHT_DECAY: 1e-5
        Weight decay to use for the policy models

    OPTIM.SPARSITY_LAMBDA: 10
        Coefficient of the sparsity incentive (reward / regularization)

    OPTIM.OVERLAP_LAMBDA: 0.0
        Whether to penalize overlap amongst masks

    ---------------------------
    (Training epoch / iteration related arguments)
    EP_IT.MAX_EPOCHS: 100
        Maximum number of epochs to train the base CNN for

    EP_IT.LOG_INTERVAL: 100
        Number of iterations within an epoch after which terminal log is displayed

    EP_IT.CKPT_STORE_INTERVAL: 100
        Number of iterations / epochs after which recurring checkpoints are stored

    ---------------------------
    (CPU / GPU Related Arguments)
    PROCESS.USE_GPU: True
        Whether to use GPU or not

    PROCESS.NUM_WORKERS: 6
        Number of workers to use for training

    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()

        self._C.HJOB = CN()
        self._C.HJOB.RANDOM_SEED = 123
        self._C.HJOB.PHASE = "aggregate_training"
        self._C.HJOB.JOB_STRING = "test"
        self._C.HJOB.WANDB_PROJECT = "DMG"
        self._C.HJOB.WANDB_DIR = "wandb_runs"

        self._C.DATA = CN()
        self._C.DATA.DATASET = "PACS"
        self._C.DATA.DOMAIN_LIST = "cartoon,photo,sketch"
        self._C.DATA.TARGET_DOMAINS = "art_painting"
        self._C.DATA.DATA_SPLIT_DIR = "data/"
        self._C.DATA.HEAD_MODE = "single"

        self._C.CKPT = CN()
        self._C.CKPT.STORAGE_DIR = "../DMG/"

        self._C.MODEL = CN()
        self._C.MODEL.BASE_MODEL = "alexnet"
        self._C.MODEL.PARAM_INIT = "custom"
        self._C.MODEL.USE_PRETRAINED = True
        self._C.MODEL.SPLIT_LAYER = "classifier.0"
        self._C.MODEL.TRAIN_FORWARD_MODE = "route"
        self._C.MODEL.EVAL_FORWARD_MODE = "route"
        self._C.MODEL.NUM_CLASSES = 7
        self._C.MODEL.MASK_LAYERS = "classifier.6"
        self._C.MODEL.POLICY_SAMPLE_MODE = "sample"
        self._C.MODEL.POLICY_CONV_MODE = False
        self._C.MODEL.MASK_INIT_SETTING = "random_uniform"
        self._C.MODEL.MASK_INIT_SCALAR = 1.0

        self._C.DATALOADER = CN()
        self._C.DATALOADER.BATCH_SIZE = 64
        self._C.DATALOADER.DATA_SAMPLING_MODE = "uniform"

        self._C.OPTIM = CN()
        self._C.OPTIM.OPTIMIZER = "Adam"
        self._C.OPTIM.LEARNING_RATE = 5e-4
        self._C.OPTIM.LEARNING_RATE_DECAY_RATE = 0.96
        self._C.OPTIM.LEARNING_RATE_DECAY_MODE = "iter"
        self._C.OPTIM.LEARNING_RATE_DECAY_STEP = 15000
        self._C.OPTIM.LEARNING_RATE_SCHEDULER = "exp"
        self._C.OPTIM.WEIGHT_DECAY = 1e-5
        self._C.OPTIM.MODEL_LEARNING_RATE = 5e-4
        self._C.OPTIM.POLICY_LEARNING_RATE = 5e-4
        self._C.OPTIM.MODEL_WEIGHT_DECAY = 1e-5
        self._C.OPTIM.POLICY_WEIGHT_DECAY = 1e-5
        self._C.OPTIM.SPARSITY_LAMBDA = 10.0
        self._C.OPTIM.OVERLAP_LAMBDA = 0.0

        self._C.EP_IT = CN()
        self._C.EP_IT.MAX_EPOCHS = 100
        self._C.EP_IT.MAX_ITER = 20000
        self._C.EP_IT.LOG_INTERVAL = 100
        self._C.EP_IT.CKPT_STORE_INTERVAL = 100

        self._C.PROCESS = CN()
        self._C.PROCESS.USE_GPU = True
        self._C.PROCESS.NUM_WORKERS = 4

        # Override parameter values from YAML file first, then from override list
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable
        self._C.freeze()

    def dump(self, file_path: str):
        """Save config at the specified file path.
        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def get_env(self):
        """
        Get a string as environment name
        based on the config attribute values
        and the phase of the job
        """
        DSET_PREFIX = ""
        ENV_NAME = ""
        # Prefix based on dataset
        if self._C.DATA.DATASET == "PACS":
            DSET_PREFIX = "pacs"
        elif self._C.DATA.DATASET == "DomainNet":
            DSET_PREFIX = "dmnt"
        else:
            print("Dataset not supported yet")

        if self._C.HJOB.PHASE == "aggregate_training":
            ENV_NAME = [self._C.HJOB.JOB_STRING, DSET_PREFIX, "AGG"]

            DOMAINS = self._C.DATA.DOMAIN_LIST.split(",")
            ENV_NAME += DOMAINS

            ENV_NAME += [
                self._C.MODEL.BASE_MODEL,
                self._C.MODEL.USE_PRETRAINED,
                self._C.MODEL.SPLIT_LAYER,
                self._C.OPTIM.OPTIMIZER,
                "LR",
                self._C.OPTIM.LEARNING_RATE,
                self._C.OPTIM.LEARNING_RATE_DECAY_RATE,
                self._C.OPTIM.LEARNING_RATE_DECAY_MODE,
                self._C.OPTIM.LEARNING_RATE_DECAY_STEP,
            ]

            if self._C.OPTIM.LEARNING_RATE_SCHEDULER != "exp":
                ENV_NAME += ["LR_SCH", self._C.OPTIM.LEARNING_RATE_SCHEDULER]

            ENV_NAME += [
                "WD",
                self._C.OPTIM.WEIGHT_DECAY,
                "BS",
                self._C.DATALOADER.BATCH_SIZE,
                self._C.DATALOADER.DATA_SAMPLING_MODE,
                "ME",
                self._C.EP_IT.MAX_EPOCHS,
            ]
        elif self._C.HJOB.PHASE == "multihead_training":
            ENV_NAME = [self._C.HJOB.JOB_STRING, DSET_PREFIX, "MH"]

            DOMAINS = self._C.DATA.DOMAIN_LIST.split(",")
            ENV_NAME += DOMAINS

            ENV_NAME += [
                self._C.MODEL.BASE_MODEL,
                self._C.MODEL.USE_PRETRAINED,
                self._C.MODEL.SPLIT_LAYER,
                "TR_FWD",
                self._C.MODEL.TRAIN_FORWARD_MODE,
                "EV_FWD",
                self._C.MODEL.EVAL_FORWARD_MODE,
                self._C.OPTIM.OPTIMIZER,
                "LR",
                self._C.OPTIM.LEARNING_RATE,
                self._C.OPTIM.LEARNING_RATE_DECAY_RATE,
                self._C.OPTIM.LEARNING_RATE_DECAY_MODE,
                self._C.OPTIM.LEARNING_RATE_DECAY_STEP,
            ]

            if self._C.OPTIM.LEARNING_RATE_SCHEDULER != "exp":
                ENV_NAME += ["LR_SCH", self._C.OPTIM.LEARNING_RATE_SCHEDULER]

            ENV_NAME += [
                "WD",
                self._C.OPTIM.WEIGHT_DECAY,
                "BS",
                self._C.DATALOADER.BATCH_SIZE,
                self._C.DATALOADER.DATA_SAMPLING_MODE,
                "ME",
                self._C.EP_IT.MAX_EPOCHS,
            ]
        elif self._C.HJOB.PHASE == "supermask_training":
            ENV_NAME = [self._C.HJOB.JOB_STRING, DSET_PREFIX, "SPMSK"]

            DOMAINS = self._C.DATA.DOMAIN_LIST.split(",")
            ENV_NAME += DOMAINS

            MASK_LAYERS = "_".join(self._C.MODEL.MASK_LAYERS.split(","))

            ENV_NAME += [
                self._C.MODEL.BASE_MODEL,
                self._C.MODEL.USE_PRETRAINED,
                MASK_LAYERS,
            ]

            ENV_NAME += [
                self._C.OPTIM.OPTIMIZER,
                "LR",
                self._C.OPTIM.MODEL_LEARNING_RATE,
                self._C.OPTIM.POLICY_LEARNING_RATE,
            ]

            if self._C.MODEL.POLICY_SAMPLE_MODE != "sample":
                ENV_NAME += ["POL_SMP", self._C.MODEL.POLICY_SAMPLE_MODE]

            if self._C.MODEL.POLICY_CONV_MODE:
                ENV_NAME += ["POL_CNV_1"]

            if self._C.OPTIM.SPARSITY_LAMBDA > 0.0:
                ENV_NAME += ["L1_SP_", self._C.OPTIM.SPARSITY_LAMBDA]

            if self._C.OPTIM.OVERLAP_LAMBDA > 0.0:
                ENV_NAME += ["IOU_OV", self._C.OPTIM.OVERLAP_LAMBDA]

            ENV_NAME += ["MSK_INIT", self._C.MODEL.MASK_INIT_SETTING]
            if self._C.MODEL.MASK_INIT_SETTING == "scalar":
                ENV_NAME += [self._C.MODEL.MASK_INIT_SCALAR]

            ENV_NAME += [
                self._C.OPTIM.LEARNING_RATE_DECAY_RATE,
                self._C.OPTIM.LEARNING_RATE_DECAY_MODE,
                self._C.OPTIM.LEARNING_RATE_DECAY_STEP,
                "WD",
                self._C.OPTIM.MODEL_WEIGHT_DECAY,
                self._C.OPTIM.POLICY_WEIGHT_DECAY,
                "BS",
                self._C.DATALOADER.BATCH_SIZE,
                self._C.DATALOADER.DATA_SAMPLING_MODE,
                "ME",
                self._C.EP_IT.MAX_EPOCHS,
            ]

        else:
            print("Job phase invalid / not supported yet")

        ENV_NAME = [str(x) for x in ENV_NAME]
        return "_".join(ENV_NAME)

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()

