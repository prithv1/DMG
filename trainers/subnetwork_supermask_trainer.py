import os
import gc
import sys
import json
import copy
import time
import wandb
import torch
import random
import argparse
import torchvision
import torch.utils.data

import numpy as np

from pprint import pprint
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision import datasets, transforms


class SubNetwork_SuperMask_Trainer:
    def __init__(
        self,
        model,
        mask_layers,
        mask_modules,
        domain_list,
        target_domains,
        train_split_obj,
        val_split_obj,
        test_split_obj,
        max_epochs,
        model_optimizer,
        mask_optimizer,
        ckpt_folder,
        params,
        model_scheduler,
        mask_scheduler,
        scheduler_mode,
        log_interval,
        viz_env_name=None,
        cuda_flag=True,
    ):
        self.model = model
        self.mask_layers = mask_layers
        self.mask_modules = mask_modules
        self.domain_list = domain_list
        self.target_domains = target_domains
        self.train_split_obj = train_split_obj
        self.val_split_obj = val_split_obj
        self.test_split_obj = test_split_obj
        self.max_epochs = max_epochs
        self.model_optimizer = model_optimizer
        self.mask_optimizer = mask_optimizer
        self.ckpt_folder = ckpt_folder
        self.params = params
        self.model_scheduler = model_scheduler
        self.mask_scheduler = mask_scheduler
        self.scheduler_mode = scheduler_mode
        self.log_interval = log_interval
        self.viz_env_name = viz_env_name
        self.cuda_flag = cuda_flag
        self.epoch = 0
        self.iteration = 0

        self.lr_decay_steps = self.params.OPTIM.LEARNING_RATE_DECAY_STEP

        # Setup wandb
        wandb.init(
            project=self.params.HJOB.WANDB_PROJECT,
            name=self.viz_env_name,
            dir=self.params.HJOB.WANDB_DIR,
        )

        # Watch model
        wandb.watch(self.model)
        for x in self.mask_modules:
            wandb.watch(x)

        # Add config
        wandb.config.update(params)

    def saveModel(
        self,
        vl_loss_perf_dict,
        vl_acc_perf_dict,
        ts_loss_perf_dict,
        ts_acc_perf_dict,
        saveFile,
    ):
        # Save the model checkpoints
        # as well as the performance dictionaries
        ckpt_dict = {
            "joint_model": self.model.joint_model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "mask_optimizer": self.mask_optimizer.state_dict(),
            "mask_layers": self.mask_layers,
            "loss_on_val": vl_loss_perf_dict,
            "acc_on_val": vl_acc_perf_dict,
            "loss_on_test": ts_loss_perf_dict,
            "acc_on_test": ts_acc_perf_dict,
            "epoch": self.epoch,
            "iteration": self.iteration,
        }

        # Add layer indices and the policy models
        for i in range(len(self.mask_layers)):
            ckpt_dict[str(self.mask_layers[i]) + "_super_mask"] = self.mask_modules[
                i
            ].state_dict()

        # Save in the specified location
        torch.save(ckpt_dict, saveFile)

    def train_epoch(self):
        since = time.time()
        running_loss = 0
        running_class_loss = 0
        running_sparsity_loss = 0
        running_corrects = 0
        running_sparsity = 0

        running_mask_ls = {x: [] for x in self.mask_layers}
        running_prob_ls = {x: [] for x in self.mask_layers}

        self.model.train()
        self.model.set_dropout_eval(True)

        # Set policy modules to train mode
        for i in range(len(self.mask_modules)):
            self.mask_modules[i].train()

        for batch_idx, (image, label, domain) in enumerate(
            self.train_split_obj.curr_loader
        ):
            label = label.long()
            if self.cuda_flag:
                image = image.cuda()
                label = label.cuda()
            iteration = (
                batch_idx + (self.epoch - 1) * len(self.train_split_obj.curr_loader) + 1
            )
            self.iteration = iteration
            self.model_optimizer.zero_grad()
            self.mask_optimizer.zero_grad()
            mask_domain = domain

            with torch.set_grad_enabled(True):
                scores, prob_ls, action_ls = self.model(
                    image,
                    self.mask_modules,
                    mask_domain,
                    self.params.MODEL.POLICY_SAMPLE_MODE,
                    self.params.MODEL.POLICY_CONV_MODE,
                )

                # Store the probabilities and the actions
                # in the specified data-structure
                for x in self.mask_layers:
                    mask_ind = self.mask_layers.index(x)
                    curr_mask = action_ls[mask_ind].mean(dim=0).cpu().detach().numpy()
                    curr_probs = prob_ls[mask_ind].mean(dim=0).cpu().detach().numpy()
                    running_mask_ls[x].append(curr_mask)
                    running_prob_ls[x].append(curr_probs)

                # Compute classification loss
                class_loss = self.model.loss_fn(scores, label)
                _, preds = torch.max(scores, 1)

                # Aggregate Sparsity Across layers
                # Also aggregate the sparsity loss
                sparsity = []
                sparsity_loss = 0
                overlap_loss = 0
                for i in range(len(self.mask_modules)):
                    sparsity.append(
                        torch.mean(self.mask_modules[i].sparsity(action_ls[i]))
                    )
                    sparsity_loss += self.mask_modules[i].sparsity_penalty()
                    overlap_loss += self.mask_modules[i].overlap_penalty()

                # Calculte total loss
                loss = (
                    class_loss.mean()
                    + self.params.OPTIM.SPARSITY_LAMBDA * sparsity_loss
                    + self.params.OPTIM.OVERLAP_LAMBDA * overlap_loss
                )

                loss.backward()
                self.model_optimizer.step()
                self.mask_optimizer.step()

                # Learning rate scheduler (per-iteration steps)
                if self.model_scheduler is not None:
                    if self.scheduler_mode == "iter":
                        self.model_scheduler.step()

                if self.mask_scheduler is not None:
                    if self.scheduler_mode == "iter":
                        self.mask_scheduler.step()

                # Show running loss
                if self.log_interval is not None:
                    if batch_idx % self.log_interval == 0:
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                self.epoch,
                                batch_idx * len(image),
                                len(self.train_split_obj.curr_loader.dataset),
                                100.0
                                * batch_idx
                                / len(self.train_split_obj.curr_loader),
                                loss.item(),
                            )
                        )

            # Keep track of statistics
            running_loss += loss.item() * image.size(0)
            running_class_loss += class_loss.mean().item() * image.size(0)
            running_sparsity_loss += sparsity_loss.mean().item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)
            running_sparsity += torch.mean(torch.stack(sparsity)) * image.size(0)

            # Get Mask Similarity Statistics
            for i in range(len(self.mask_layers)):
                iou_overlap_dict = self.mask_modules[i].mask_overlap(
                    self.mask_layers[i]
                )

            # Log per-iteration data in wandb
            wandb.log({"Iter Train Loss": loss.data.item()}, step=self.iteration)
            wandb.log(
                {"Iter Train CLoss": class_loss.mean().data.item()}, step=self.iteration
            )
            wandb.log(
                {"Iter Train SLoss": sparsity_loss.mean().data.item()},
                step=self.iteration,
            )
            wandb.log(
                {
                    "Iter Train CAcc": torch.sum(preds == label.data)
                    .double()
                    .data.item()
                    / image.size(0)
                },
                step=self.iteration,
            )
            wandb.log(
                {"Iter Train Sparsity": torch.mean(torch.stack(sparsity)).data.item()},
                step=self.iteration,
            )
        epoch_loss = running_loss / len(self.train_split_obj.curr_loader.dataset)
        epoch_class_loss = running_class_loss / len(
            self.train_split_obj.curr_loader.dataset
        )
        epoch_sparsity_loss = running_sparsity_loss / len(
            self.train_split_obj.curr_loader.dataset
        )
        epoch_acc = running_corrects.double() / len(
            self.train_split_obj.curr_loader.dataset
        )
        epoch_sparsity = running_sparsity.double() / len(
            self.train_split_obj.curr_loader.dataset
        )

        # Log per-iteration data in wandb
        wandb.log({"Epoch Train Loss": epoch_loss}, step=self.iteration)
        wandb.log({"Epoch Train CLoss": epoch_class_loss}, step=self.iteration)
        wandb.log({"Epoch Train SLoss": epoch_sparsity_loss}, step=self.iteration)
        wandb.log({"Epoch Train CAcc": epoch_acc}, step=self.iteration)
        wandb.log({"Epoch Train Sparsity": epoch_sparsity}, step=self.iteration)
        time_elapsed = time.time() - since

    def evaluate(self, phase="val"):
        since = time.time()
        sparsity_stats = [[] for x in range(len(self.mask_modules))]
        running_loss, running_corrects = 0, 0
        self.model.eval()
        for i in range(len(self.mask_modules)):
            self.mask_modules[i].eval()

        if phase == "val":
            eval_loader = self.val_split_obj.curr_loader
        elif phase == "test":
            eval_loader = self.test_split_obj.curr_loader
        else:
            print("Phase not supported for evaluation")

        for batch_idx, (image, label, domain) in enumerate(eval_loader):
            label = label.long()
            if self.cuda_flag:
                image = image.cuda()
                label = label.cuda()

            if phase == "val":
                policy_domain = domain
                with torch.set_grad_enabled(False):
                    scores, _, action_ls = self.model(
                        image,
                        self.mask_modules,
                        policy_domain,
                        "greedy",
                        self.params.MODEL.POLICY_CONV_MODE,
                    )

                    for i in range(len(self.mask_modules)):
                        sparsity_stats[i].append(
                            self.mask_modules[i].sparsity(action_ls[i]).mean().item()
                        )

            elif phase == "test":
                scores = []
                for eval_domain in self.domain_list:
                    policy_domain = [eval_domain] * len(domain)
                    with torch.set_grad_enabled(False):
                        score, _, action_ls = self.model(
                            image,
                            self.mask_modules,
                            policy_domain,
                            "softscale",
                            self.params.MODEL.POLICY_CONV_MODE,
                        )
                        scores.append(score)
                        for i in range(len(self.mask_modules)):
                            sparsity_stats[i].append(
                                self.mask_modules[i]
                                .sparsity(action_ls[i])
                                .mean()
                                .item()
                            )
                scores = torch.stack(scores)
                scores = scores.mean(0)

            _, preds = torch.max(scores, 1)
            eval_loss = self.model.loss_fn(scores, label)
            eval_loss = eval_loss.mean()
            running_loss += eval_loss.item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)

        time_elapsed = time.time() - since
        epoch_loss = running_loss / len(eval_loader.dataset)
        epoch_acc = running_corrects.double() / len(eval_loader.dataset)
        sparsity_stats = np.array(sparsity_stats)
        sparsity_stats = np.mean(sparsity_stats, axis=1).tolist()
        return epoch_loss, epoch_acc, sparsity_stats

    def train(self, ckpt_store_interval=20):
        ckpt_store_int = ckpt_store_interval
        running_vl_loss_dict = {x: 0 for x in self.domain_list}
        running_vl_acc_dict = {x: 0 for x in self.domain_list}

        running_vl_loss_dict["overall"] = 0
        running_vl_acc_dict["overall"] = 0

        running_ts_loss_dict = {x: 0 for x in self.target_domains}
        running_ts_acc_dict = {x: 0 for x in self.target_domains}

        running_ts_loss_dict["overall"] = 0
        running_ts_acc_dict["overall"] = 0

        vl_neurons_activated = []
        for i in range(len(self.mask_layers)):
            neuron_stats = {x: 0 for x in self.domain_list}
            vl_neurons_activated.append(neuron_stats)

        ts_neurons_activated = []
        for i in range(len(self.mask_layers)):
            neuron_stats = {x: 0 for x in self.target_domains}
            ts_neurons_activated.append(neuron_stats)

        best_score = None
        last_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()

            wandb.log(
                {"Iteration": self.iteration, "Epoch": self.epoch}, step=self.iteration
            )

            # Performance on the val dataset
            for domain in self.domain_list:
                self.val_split_obj.set_domain_spec_mode(True, domain)
                temp_loss, temp_acc, temp_sparsity = self.evaluate("val")
                running_vl_loss_dict[domain] = temp_loss
                running_vl_acc_dict[domain] = temp_acc.data.item()
                for j in range(len(self.mask_layers)):
                    vl_neurons_activated[j][domain] = temp_sparsity[j]
                    wandb.log(
                        {
                            domain
                            + " : Blocks Activated at "
                            + self.mask_layers[j]: temp_sparsity[j]
                        },
                        step=self.iteration,
                    )

                wandb.log(
                    {domain + " : Validation Loss": temp_loss}, step=self.iteration
                )
                wandb.log(
                    {domain + " : Validation Accuracy": temp_acc.data.item()},
                    step=self.iteration,
                )
                self.val_split_obj.set_domain_spec_mode(False)

            vl_loss = np.mean([running_vl_loss_dict[x] for x in self.domain_list])
            vl_acc = np.mean([running_vl_acc_dict[x] for x in self.domain_list])
            running_vl_loss_dict["overall"] = vl_loss
            running_vl_acc_dict["overall"] = vl_acc
            wandb.log(
                {"Overall : Validation Loss": running_vl_loss_dict["overall"]},
                step=self.iteration,
            )
            wandb.log(
                {"Overall : Validation Accuracy": running_vl_acc_dict["overall"]},
                step=self.iteration,
            )

            print("-----------------------------------")
            print("Fine-grained validation performance")
            print("-----------------------------------")
            print("Loss")
            pprint(running_vl_loss_dict)
            print("Accuracy")
            pprint(running_vl_acc_dict)
            print("-----------------------------------")

            # Performance on the test dataset
            for domain in self.target_domains:
                self.test_split_obj.set_domain_spec_mode(True, domain)
                temp_loss, temp_acc, temp_sparsity = self.evaluate("test")
                running_ts_loss_dict[domain] = temp_loss
                running_ts_acc_dict[domain] = temp_acc.data.item()
                for j in range(len(self.mask_layers)):
                    ts_neurons_activated[j][domain] = temp_sparsity[j]
                    wandb.log(
                        {
                            domain
                            + " : Blocks Activated at "
                            + self.mask_layers[j]: temp_sparsity[j]
                        },
                        step=self.iteration,
                    )

                wandb.log({domain + " : Test Loss": temp_loss}, step=self.iteration)
                wandb.log(
                    {domain + " : Test Accuracy": temp_acc.data.item()},
                    step=self.iteration,
                )
                self.test_split_obj.set_domain_spec_mode(False)

            ts_loss = np.mean([running_ts_loss_dict[x] for x in self.target_domains])
            ts_acc = np.mean([running_ts_acc_dict[x] for x in self.target_domains])
            running_ts_loss_dict["overall"] = ts_loss
            running_ts_acc_dict["overall"] = ts_acc
            wandb.log(
                {"Overall : Test Loss": running_ts_loss_dict["overall"]},
                step=self.iteration,
            )
            wandb.log(
                {"Overall : Test Accuracy": running_ts_acc_dict["overall"]},
                step=self.iteration,
            )

            print("-----------------------------------")
            print("Fine-grained test performance")
            print("-----------------------------------")
            print("Loss")
            pprint(running_ts_loss_dict)
            print("Accuracy")
            pprint(running_ts_acc_dict)
            print("-----------------------------------")

            # Store checkpoints
            if self.epoch % ckpt_store_int == 0 or self.epoch == 0:
                self.saveModel(
                    running_vl_loss_dict,
                    running_vl_acc_dict,
                    running_ts_loss_dict,
                    running_ts_acc_dict,
                    self.ckpt_folder + "/model_ep_" + str(self.epoch) + ".pth",
                )

            score = vl_acc
            if best_score is None or score > best_score:
                best_score = score
                self.saveModel(
                    running_vl_loss_dict,
                    running_vl_acc_dict,
                    running_ts_loss_dict,
                    running_ts_acc_dict,
                    self.ckpt_folder + "/best_so_far.pth",
                )
                with open(self.ckpt_folder + "/val_loss.json", "w") as f:
                    json.dump(running_vl_loss_dict, f)
                with open(self.ckpt_folder + "/val_acc.json", "w") as f:
                    json.dump(running_vl_acc_dict, f)
                with open(self.ckpt_folder + "/test_loss.json", "w") as f:
                    json.dump(running_ts_loss_dict, f)
                with open(self.ckpt_folder + "/test_acc.json", "w") as f:
                    json.dump(running_ts_acc_dict, f)

                # Store best values in tables
                best_vl_score_wandb_table = wandb.Table(
                    columns=["Domain", "Validation Loss", "Validation Accuracy"]
                )
                for key, val in running_vl_loss_dict.items():
                    best_vl_score_wandb_table.add_data(
                        key,
                        str(running_vl_loss_dict[key]),
                        str(running_vl_acc_dict[key]),
                    )
                wandb.log({"Best_Validation_Score_Table": best_vl_score_wandb_table})

                # Store test values in tables
                best_ts_score_wandb_table = wandb.Table(
                    columns=["Domain", "Test Loss", "Test Accuracy"]
                )
                for key, val in running_ts_loss_dict.items():
                    best_ts_score_wandb_table.add_data(
                        key,
                        str(running_ts_loss_dict[key]),
                        str(running_ts_acc_dict[key]),
                    )
                wandb.log({"Best_Test_Score_Table": best_ts_score_wandb_table})

            # Check for epoch level scheduler step
            if self.model_scheduler is not None:
                if self.scheduler_mode == "epoch":
                    if self.params.OPTIM.LEARNING_RATE_SCHEDULER == "exp":
                        self.model_scheduler.step()
                    elif self.params.OPTIM.LEARNING_RATE_SCHEDULER == "invlr":
                        self.model_scheduler.step()

            # if self.mask_scheduler is not None:
            #     if self.scheduler_mode == "epoch":
            #         if self.params.OPTIM.LEARNING_RATE_SCHEDULER == "exp":
            #             self.mask_scheduler.step()
            #         elif self.params.OPTIM.LEARNING_RATE_SCHEDULER == "invlr":
            #             self.mask_scheduler.step()
