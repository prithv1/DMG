import os
import sys
import torch
import torchvision
import torch.utils.data

import numpy as np

from pprint import pprint
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms

ALEXNET_LAYERS = ["classifier.1", "classifier.4", "classifier.6"]
RESNET_LAYERS = ["layer3", "layer4", "fc"]

ALEXNET_DROPOUT_LAYERS = ["classifier.0", "classifier.3"]

# Specify the sizes of the input
# activations for each of the valid masking layers
RESNET18_LAYER_SIZES = {"layer3": 128 * 28 * 28, "layer4": 256 * 14 * 14, "fc": 512}

RESNET50_LAYER_SIZES = {"layer3": 512 * 28 * 28, "layer4": 1024 * 14 * 14, "fc": 2048}

RESNET18_CNV_LAYER_SIZES = {"layer3": 128, "layer4": 256, "fc": 512}

RESNET50_CNV_LAYER_SIZES = {"layer3": 512, "layer4": 1024, "fc": 2048}


class SubNetwork_SuperMask_Model(nn.Module):
    def __init__(self, mask_layers, joint_model):
        super(SubNetwork_SuperMask_Model, self).__init__()

        # Mask-Layers can be provided as
        # [classifier.4, classifier.6]
        self.mask_layers = mask_layers

        # The actual multi-head model
        self.joint_model = joint_model

        # Store base model in a variable
        self.base_model = self.joint_model.model_name

        # List legal mask areas
        self.legal_mask_areas = {
            "alexnet": ALEXNET_LAYERS,
            "resnet18": RESNET_LAYERS,
            "resnet50": RESNET_LAYERS,
        }

        # We'll go over the model structure and identify
        # legal mask / regions -- layers at whose input
        # activations it is legal to apply the mask
        self.legal_mask_areas, self.mask_areas = {}, {}

        # Define loss
        # The `input` is expected to contain raw, unnormalized scores for each class and associated label.
        self.criterion = nn.CrossEntropyLoss()

    def get_mask_struct(self, conv_mode=False):
        # Retrieve the input activation sizes
        act_size = []
        if self.base_model in ["alexnet", "vgg16", "vgg16_bn"]:
            for mask_layer in self.mask_layers:
                if "classifier" in mask_layer:
                    act_size.append(
                        self.joint_model.model_ft.classifier[
                            int(mask_layer.split(".")[-1])
                        ].in_features
                    )
                else:
                    print("Masking this layer is not supported yet")
        elif self.base_model == "resnet18":
            for mask_layer in self.mask_layers:
                if mask_layer == "layer3":
                    if conv_mode:
                        act_size.append(RESNET18_CNV_LAYER_SIZES[mask_layer])
                    else:
                        act_size.append(RESNET18_LAYER_SIZES[mask_layer])
                elif mask_layer == "layer4":
                    if conv_mode:
                        act_size.append(RESNET18_CNV_LAYER_SIZES[mask_layer])
                    else:
                        act_size.append(RESNET18_LAYER_SIZES[mask_layer])
                elif mask_layer == "fc":
                    if conv_mode:
                        act_size.append(RESNET18_CNV_LAYER_SIZES[mask_layer])
                    else:
                        act_size.append(RESNET18_LAYER_SIZES[mask_layer])
        elif self.base_model == "resnet50":
            for mask_layer in self.mask_layers:
                if mask_layer == "layer3":
                    if conv_mode:
                        act_size.append(RESNET50_CNV_LAYER_SIZES[mask_layer])
                    else:
                        act_size.append(RESNET50_LAYER_SIZES[mask_layer])
                elif mask_layer == "layer4":
                    if conv_mode:
                        act_size.append(RESNET50_CNV_LAYER_SIZES[mask_layer])
                    else:
                        act_size.append(RESNET50_LAYER_SIZES[mask_layer])
                elif mask_layer == "fc":
                    if conv_mode:
                        act_size.append(RESNET50_CNV_LAYER_SIZES[mask_layer])
                    else:
                        act_size.append(RESNET50_LAYER_SIZES[mask_layer])
        else:
            print("Model not supported yet")
        return act_size

    def set_dropout_eval(self, flag=True):
        # Set the dropout to eval mode for the
        # specified networks -- only alexnet, vgg16
        if self.base_model == "alexnet":
            dropout_layers = ALEXNET_DROPOUT_LAYERS
            for dropout_layer in dropout_layers:
                if flag:
                    self.joint_model.model_ft.classifier[
                        int(dropout_layer.split(".")[-1])
                    ].eval()
                else:
                    self.joint_model.model_ft.classifier[
                        int(dropout_layer.split(".")[-1])
                    ].train()
        elif self.base_model == "resnet18":
            pass
        elif self.base_model == "resnet50":
            pass
        else:
            print("Base model not supported yet")

    def classifier_dom_mask_forward(
        self, img, policy_modules, policy_domain, mode="sample"
    ):
        # This is only for alexnet and vgg-16 based architectures
        # We don't need these for resnet-18 and 50
        prob_ls, action_ls, logProb_ls = [], [], []

        # Extract features
        feats = self.joint_model.model_ft.features(img)
        feats = self.joint_model.model_ft.avgpool(feats)
        feats = feats.view(feats.size(0), -1)
        # Go through the classifier
        for j in range(len(self.joint_model.model_ft.classifier)):
            if "classifier." + str(j) in self.mask_layers:
                rel_ind = self.mask_layers.index("classifier." + str(j))
                feats, action, probs = policy_modules[rel_ind](
                    feats, policy_domain, mode
                )
                prob_ls.append(probs)
                action_ls.append(action)
                feats = self.joint_model.model_ft.classifier[j](feats)
            else:
                feats = self.joint_model.model_ft.classifier[j](feats)

        scores = feats
        return scores, prob_ls, action_ls, logProb_ls

    def forward(
        self, img, policy_modules, policy_domain, mode="sample", conv_mode=False
    ):

        # Check if the number of layer indices match
        # match the number of policy modules
        assert len(self.mask_layers) == len(
            policy_modules
        ), "Unequal number of layers and modules"

        # Data structures to store the scores,
        # probabilities, actions and logProbs
        prob_ls, action_ls, logProb_ls = [], [], []

        if self.base_model == "alexnet":
            scores, prob_ls, action_ls, logProb_ls = self.classifier_dom_mask_forward(
                img, policy_modules, policy_domain, mode
            )
        elif self.base_model in ["resnet18", "resnet50"]:

            # Forward pass based on the specified masking layers
            feats = self.joint_model.model_ft.conv1(img)
            feats = self.joint_model.model_ft.bn1(feats)
            feats = self.joint_model.model_ft.relu(feats)
            feats = self.joint_model.model_ft.maxpool(feats)
            feats = self.joint_model.model_ft.layer1(feats)
            feats = self.joint_model.model_ft.layer2(feats)

            if "layer3" in self.mask_layers:
                feat_shape = feats.shape
                pol_ind = self.mask_layers.index("layer3")
                if not conv_mode:
                    feats = feats.view(feat_shape[0], -1)
                feats, action, probs = policy_modules[pol_ind](
                    feats, policy_domain, mode, conv_mode
                )
                prob_ls.append(probs)
                action_ls.append(action)
                feats = feats.view(feat_shape)
                feats = self.joint_model.model_ft.layer3(feats)
            else:
                feats = self.joint_model.model_ft.layer3(feats)

            if "layer4" in self.mask_layers:
                feat_shape = feats.shape
                pol_ind = self.mask_layers.index("layer4")
                if not conv_mode:
                    feats = feats.view(feat_shape[0], -1)
                feats, action, probs = policy_modules[pol_ind](
                    feats, policy_domain, mode, conv_mode
                )
                prob_ls.append(probs)
                action_ls.append(action)
                feats = feats.view(feat_shape)
                feats = self.joint_model.model_ft.layer4(feats)
            else:
                feats = self.joint_model.model_ft.layer4(feats)

            feats = self.joint_model.model_ft.avgpool(feats)

            feats = feats.view(feats.size(0), -1)
            if "fc" in self.mask_layers:
                pol_ind = self.mask_layers.index("fc")
                feats, action, probs = policy_modules[pol_ind](
                    feats, policy_domain, mode, conv_mode
                )
                prob_ls.append(probs)
                action_ls.append(action)
            scores = self.joint_model.model_ft.fc(feats)
            del feats, probs, action
        else:
            print("Model not supported yet")
        return scores, prob_ls, action_ls

    def loss_fn(self, scores, labels):
        # Basic cross-entropy criterion
        # loss_val = self.criterion(scores, labels)
        loss_val = F.cross_entropy(scores, labels, reduction="none")
        return loss_val

    @classmethod
    def from_config(cls, config, joint_model, mask_layers):
        _C = config
        return cls(mask_layers=mask_layers, joint_model=joint_model)

