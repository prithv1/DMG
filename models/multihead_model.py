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

from pprint import pprint
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms

# Possible split layers for different model architectures
alexnet_task_net_layer = ["classifier." + str(x) for x in range(0, 7)]
vgg16_task_net_layers = ["classifier." + str(x) for x in range(0, 7)]
vgg16_bn_task_net_layers = ["classifier." + str(x) for x in range(0, 7)]
resnet18_task_net_layers = ["fc"]
resnet50_task_net_layers = ["fc"]

ALEXNET_POSS_SPLIT_LAYERS = [
    "classifier.0",  # the entire network post pool5
    "classifier.1",  # Following linear layer
    "classifier.4",  # Following linear layer
    "classifier.6",  # Last linear layer
]

RESNET_18_POSS_SPLIT_LAYERS = ["fc"]  # Only the last classifier layer

RESNET_50_POSS_SPLIT_LAYERS = ["fc"]  # Only the last classifier layer

# Identity class
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MultiHead_Model(nn.Module):
    def __init__(
        self,
        domain_list,
        model_name,
        num_classes,
        task_net_layer=None,
        init_setting="custom",
        use_pretrained=True,
    ):
        super(MultiHead_Model, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.domain_list = domain_list
        self.task_net_layer = task_net_layer

        self.criterion = nn.CrossEntropyLoss()

        self.input_size = 0

        # Define model structure based on the specified arguments
        # This decides the feature-network and task-network splits
        self.model_fn = getattr(models, self.model_name)
        self.model_ft = self.model_fn(pretrained=self.use_pretrained)
        if self.model_name == "alexnet":
            num_feats = self.model_ft.classifier[6].in_features
            self.input_size = 224
            self.model_ft.classifier[6] = nn.Linear(num_feats, self.num_classes)
            if init_setting == "custom":
                self.model_ft.classifier[6].apply(weights_init)

            # Get a list of classifier layers
            classifier_layers = ["classifier." + str(x) for x in range(0, 7)]

            # Identify which classifier layer is being split at
            classifier_ind = int(self.task_net_layer.split(".")[-1])

            # Get the whole module list
            module_list = list(
                self.model_ft.classifier[
                    classifier_ind : len(classifier_layers)
                ].children()
            )

            # Create task networks for every domain
            self.domain_task_nets = nn.ModuleDict(
                {x: nn.Sequential(*module_list) for x in self.domain_list}
            )

            # Make older versions identity
            for i in range(classifier_ind, 7):
                self.model_ft.classifier[i] = Identity()

        elif "resnet" in self.model_name:
            num_feats = self.model_ft.fc.in_features
            self.input_size = 224

            # Create task networks
            self.domain_task_nets = nn.ModuleDict(
                {
                    x: nn.Sequential(nn.Linear(num_feats, self.num_classes))
                    for x in self.domain_list
                }
            )

            # Weight and bias initialization
            if init_setting == "custom":
                for x in self.domain_list:
                    self.domain_task_nets[x][0].apply(weights_init)

            self.model_ft.fc = nn.Identity()
        else:
            print("Model type not supported")

    # Note that our forward pass
    # has to be aware of the domain-ID
    # It has to route examples to specific
    # domain task networks accordingly
    def forward(self, img, dom):
        # Extract features first
        feats = self.model_ft(img)
        # Convert mini-batch domain list to list of indices
        scores = []
        for i in range(len(dom)):
            scores.append(self.domain_task_nets[dom[i]](torch.unsqueeze(feats[i], 0)))
        scores = torch.cat(scores)
        return scores

    def loss_fn(self, scores, labels):
        # Basic cross-entropy criterion
        loss_val = self.criterion(scores, labels)
        return loss_val

    def loss_gpu(self, flag=True):
        if flag:
            self.criterion.cuda()

    # Now we need another evaluation time forward mode
    # We average the raw un-normalized scores
    # from all the heads. One instance -> through all heads
    def avg_forward(self, img):
        # Extract features
        feats = self.model_ft(img)
        # Extract scores from all the
        # domain-specific task networks
        scores = []
        for dom in self.domain_list:
            scores.append(self.domain_task_nets[dom](feats))

        return torch.mean(torch.stack(scores), dim=0)

    # Average probability and then forward
    def avg_prob_forward(self, img):
        # Extract features
        feats = self.model_ft(img)
        # Extract scores from all the
        # domain-specific task networks
        scores = []
        for dom in self.domain_list:
            scores.append(nn.Softmax(dim=1)(self.domain_task_nets[dom](feats)))

        return torch.mean(torch.stack(scores), dim=0)

    @classmethod
    def from_config(cls, config):
        _C = config
        domains = _C.DATA.DOMAIN_LIST
        if "," in domains:
            domains = _C.DATA.DOMAIN_LIST.split(",")
        return cls(
            domain_list=domains,
            model_name=_C.MODEL.BASE_MODEL,
            num_classes=_C.MODEL.NUM_CLASSES,
            task_net_layer=_C.MODEL.SPLIT_LAYER,
            init_setting=_C.MODEL.PARAM_INIT,
            use_pretrained=_C.MODEL.USE_PRETRAINED,
        )
