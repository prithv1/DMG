"""
Reference: https://github.com/pytorch/tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py
"""
import os
import sys
import torch
import torchvision
import torch.utils.data

import numpy as np

from utils.misc import weights_init

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms

ALEXNET_DROPOUT_LAYERS = ["classifier.0", "classifier.3"]
VGG16_DROPOUT_LAYERS = ["classifier.2", "classifier.5"]


class Basic_Model(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        split_layer,
        init_setting="custom",
        use_pretrained=True,
    ):
        super(Basic_Model, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.split_layer = split_layer
        self.use_pretrained = use_pretrained

        self.criterion = nn.CrossEntropyLoss()

        self.input_size = 0

        self.model_fn = getattr(models, self.model_name)
        self.model_ft = self.model_fn(pretrained=self.use_pretrained)

        # Iterate over specific model types
        if self.model_name == "alexnet":
            num_feats = self.model_ft.classifier[6].in_features
            self.input_size = 224
            self.model_ft.classifier[6] = nn.Linear(num_feats, self.num_classes)
            if init_setting == "custom":
                self.model_ft.classifier[6].apply(weights_init)
            print(self.model_ft.classifier[6].weight)
            print(self.model_ft.classifier[6].bias)
        elif "resnet" in self.model_name:
            num_feats = self.model_ft.fc.in_features
            self.input_size = 224
            self.model_ft.fc = nn.Linear(num_feats, self.num_classes)
            if init_setting == "custom":
                self.model_ft.fc.apply(weights_init)
        else:
            print("Model type not supported yet")

    def forward(self, img):
        scores = self.model_ft(img)
        return scores

    def loss_fn(self, scores, labels):
        loss_val = self.criterion(scores, labels)
        return loss_val

    def loss_gpu(self, flag=True):
        if flag:
            self.criterion.cuda()

    @classmethod
    def from_config(cls, config):
        _C = config
        return cls(
            model_name=_C.MODEL.BASE_MODEL,
            num_classes=_C.MODEL.NUM_CLASSES,
            split_layer=_C.MODEL.SPLIT_LAYER,
            init_setting=_C.MODEL.PARAM_INIT,
            use_pretrained=_C.MODEL.USE_PRETRAINED,
        )
