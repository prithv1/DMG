import os
import sys
import torch
import torchvision
import torch.utils.data

import numpy as np

from pprint import pprint
from itertools import combinations

from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Bernoulli, RelaxedBernoulli
from torchvision import datasets, models, transforms

SMOOTH = 1e-6


class SuperMask(nn.Module):
    def __init__(self, domain_list, act_size, init_setting="random", init_scalar=1):
        super(SuperMask, self).__init__()
        self.domain_list = domain_list
        self.act_size = act_size
        self.init_setting = init_setting
        self.init_scalar = init_scalar

        # Define the super mask logits
        if self.init_setting == "random_uniform":
            self.super_mask_logits = nn.ParameterDict(
                {
                    x: nn.Parameter(torch.rand(self.act_size, requires_grad=True))
                    for x in self.domain_list
                }
            )
        elif self.init_setting == "scalar":
            param_tensor = torch.ones(self.act_size, requires_grad=True)
            param_tensor = param_tensor.new_tensor(
                [self.init_scalar] * self.act_size, requires_grad=True
            )
            self.super_mask_logits = nn.ParameterDict(
                {x: nn.Parameter(param_tensor.clone()) for x in self.domain_list}
            )

    def forward(self, activation, domain, mode="sample", conv_mode=False):
        # Mask repeated along channel dimensions if conv_mode == True
        probs = [nn.Sigmoid()(self.super_mask_logits[x]) for x in domain]
        probs = torch.stack(probs)
        if mode == "sample":
            mask_dist = Bernoulli(probs)
            hard_mask = mask_dist.sample()
            soft_mask = probs
            mask = (hard_mask - soft_mask).detach() + soft_mask
            if conv_mode and len(activation.shape) > 2:
                apply_mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = mask * activation
        elif mode == "greedy":
            hard_mask = (probs > 0.5).float()
            soft_mask = probs
            mask = (hard_mask - soft_mask).detach() + soft_mask
            if conv_mode and len(activation.shape) > 2:
                apply_mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = mask * activation
        elif mode == "softscale":
            hard_mask = (probs > 0.5).float()
            soft_mask = probs
            mask = hard_mask
            if conv_mode and len(activation.shape) > 2:
                apply_mask = soft_mask.view(
                    soft_mask.shape[0], soft_mask.shape[1], 1, 1
                )
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = soft_mask * activation
        elif mode == "avg_mask_softscale":
            # Average all the source domain masks
            # instead of combining them
            all_probs = [
                nn.Sigmoid()(self.super_mask_logits[x]) for x in self.domain_list
            ]
            all_probs = torch.mean(torch.stack(all_probs), 0)
            mean_mask = [all_probs for x in domain]
            mean_mask = torch.stack(mean_mask)
            soft_mask = mean_mask
            hard_mask = (mean_mask > 0.5).float()
            mask = hard_mask
            if conv_mode and len(activation.shape) > 2:
                apply_mask = soft_mask.view(
                    soft_mask.shape[0], soft_mask.shape[1], 1, 1
                )
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = soft_mask * activation

        return (activation, mask, soft_mask)

    def sparsity(self, mask):
        return torch.mean(mask, dim=1)

    def sparsity_penalty(self):
        sparse_pen = 0
        for _, v in self.super_mask_logits.items():
            sparse_pen += torch.sum(nn.Sigmoid()(v))
        return sparse_pen

    def overlap_penalty(self):
        overlap_pen = 0
        domain_pairs = list(combinations(self.domain_list, 2))
        for pair in domain_pairs:
            dom1, dom2 = pair
            mask1 = nn.Sigmoid()(self.super_mask_logits[dom1])
            mask2 = nn.Sigmoid()(self.super_mask_logits[dom2])
            intersection = torch.sum(mask1 * mask2)
            union = torch.sum(mask1 + mask2 - mask1 * mask2)
            iou = (intersection + SMOOTH) / (union + SMOOTH)
            overlap_pen += iou
        overlap_pen /= len(domain_pairs)
        return overlap_pen

    def mask_overlap(self, layer_name=""):
        if layer_name != "":
            prefix = layer_name + " : "
        else:
            prefix = ""
        domain_pairs = combinations(self.domain_list, 2)
        iou_overlap_dict = {}
        for pair in domain_pairs:
            mask_0 = nn.Sigmoid()(self.super_mask_logits[pair[0]])
            mask_1 = nn.Sigmoid()(self.super_mask_logits[pair[1]])
            mask_0 = mask_0 > 0.5
            mask_1 = mask_1 > 0.5
            intersection = (mask_0 & mask_1).float().sum()
            union = (mask_0 | mask_1).float().sum()
            iou = (intersection + SMOOTH) / (union + SMOOTH)
            iou_overlap_dict[
                prefix + pair[0] + ", " + pair[1] + " IoU-Ov"
            ] = iou.data.item()
        iou_overlap_dict[prefix + "overall IoU-Ov"] = np.mean(
            [x for x in list(iou_overlap_dict.values())]
        )
        return iou_overlap_dict

    @classmethod
    def from_config(cls, config, act_size):
        _C = config
        domains = _C.DATA.DOMAIN_LIST
        if "," in domains:
            domains = _C.DATA.DOMAIN_LIST.split(",")
        return cls(
            domains, act_size, _C.MODEL.MASK_INIT_SETTING, _C.MODEL.MASK_INIT_SCALAR
        )
