import os
import sys
import json
import copy
import time
import torch
import wandb
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


class MultiHead_Trainer:
    def __init__(
        self,
        model,
        cuda_flag,
        train_split_obj,
        val_split_obj,
        test_split_obj,
        train_forward_mode,
        eval_forward_mode,
        max_epochs,
        optimizer,
        viz_env_name,
        log_interval,
        scheduler,
        scheduler_mode,
        lr_decay_steps,
        ckpt_folder,
        params,
    ):
        self.model = model
        self.cuda_flag = cuda_flag
        self.train_split_obj = train_split_obj
        self.val_split_obj = val_split_obj
        self.test_split_obj = test_split_obj
        self.train_forward_mode = train_forward_mode
        self.eval_forward_mode = eval_forward_mode
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.viz_env_name = viz_env_name
        self.log_interval = log_interval
        self.scheduler = scheduler
        self.scheduler_mode = scheduler_mode
        self.lr_decay_steps = lr_decay_steps
        self.ckpt_folder = ckpt_folder
        self.params = params
        self.epoch = 0
        self.iteration = 0

        self.target_domains = self.params.DATA.TARGET_DOMAINS.split(",")

        # Setup wandb
        wandb.init(
            project=self.params.HJOB.WANDB_PROJECT,
            name=self.viz_env_name,
            dir=self.params.HJOB.WANDB_DIR,
        )

        # Watch model
        wandb.watch(self.model)

        # Add config
        wandb.config.update(params)

    def saveModel(self, saveFile, params):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "iteration": self.iteration,
            },
            saveFile,
        )

    def train_epoch(self):
        self.model.train()
        since = time.time()
        running_loss, running_corrects = 0, 0
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
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                if self.train_forward_mode == "route":
                    outputs = self.model(image, domain)
                elif self.train_forward_mode == "avg_score_fwd":
                    outputs = self.model.avg_forward(image)
                else:
                    print("Cannot forward pass in this mode")
                loss = self.model.loss_fn(outputs, label)
                wandb.log({"Training Iter Loss": loss.data.item()}, step=self.iteration)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    if self.scheduler_mode == "iter":
                        self.scheduler.step()

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

            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)
        time_elapsed = time.time() - since
        wandb.log({"Train Epoch Time": time_elapsed}, step=self.iteration)
        epoch_loss = running_loss / len(self.train_split_obj.curr_loader.dataset)
        epoch_acc = running_corrects.double() / len(
            self.train_split_obj.curr_loader.dataset
        )
        return epoch_loss, epoch_acc

    def validate(self, mode="val"):
        self.model.eval()
        since = time.time()
        running_loss, running_corrects = 0, 0
        if mode == "val":
            rel_loader = self.val_split_obj.curr_loader
        elif mode == "test":
            rel_loader = self.test_split_obj.curr_loader
        else:
            print("Split mode not supported yet")

        for batch_idx, (image, label, domain) in enumerate(rel_loader):
            label = label.long()
            if self.cuda_flag:
                image = image.cuda()
                label = label.cuda()

            with torch.set_grad_enabled(False):
                if mode == "val":
                    if self.eval_forward_mode == "route":
                        outputs = self.model(image, domain)
                    elif self.eval_forward_mode == "avg_score_fwd":
                        outputs = self.model.avg_forward(image)
                    elif self.eval_forward_mode == "avg_prob_fwd":
                        outputs = self.model.avg_prob_forward(image)
                    else:
                        print("Eval forward mode not identified")
                elif mode == "test":
                    outputs = self.model.avg_forward(image)
                else:
                    print("Split mode not supported yet")

            loss = self.model.loss_fn(outputs, label)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)

        time_elapsed = time.time() - since
        wandb.log({"Val Epoch Time": time_elapsed}, step=self.iteration)
        epoch_loss = running_loss / len(rel_loader.dataset)
        epoch_acc = running_corrects.double() / len(rel_loader.dataset)
        return epoch_loss, epoch_acc

    def train(self, domain_list=None, ckpt_store_interval=20):
        # Checkpoint storing interval
        ckpt_store_int = ckpt_store_interval
        # To keep track of running performance
        running_vl_loss_dict = {x: 0 for x in domain_list}
        running_vl_acc_dict = {x: 0 for x in domain_list}
        running_vl_loss_dict["overall"] = 0
        running_vl_acc_dict["overall"] = 0

        running_ts_loss_dict = {x: 0 for x in self.target_domains}
        running_ts_acc_dict = {x: 0 for x in self.target_domains}
        running_ts_loss_dict["overall"] = 0
        running_ts_acc_dict["overall"] = 0

        # Train and repeat the above functions
        # with appropriate logging
        best_score = None
        last_epoch = 0

        # We don't need to specify which set of
        # parameters to keep gradients on for
        for epoch in range(1, self.max_epochs + 1):
            # Set global epoch
            self.epoch = epoch
            self.train_epoch()

            # Plot iteration versus epochs
            wandb.log(
                {"Iteration": self.iteration, "Epoch": self.epoch}, step=self.iteration
            )

            # Evaluate on the val split
            for domain in domain_list:
                self.val_split_obj.set_domain_spec_mode(True, domain)
                temp_loss, temp_acc = self.validate("val")
                running_vl_loss_dict[domain] = temp_loss
                running_vl_acc_dict[domain] = temp_acc.data.item()
                self.val_split_obj.set_domain_spec_mode(False)

            # Calculate overall performance
            # Validation
            vl_loss = np.mean([running_vl_loss_dict[x] for x in domain_list])
            vl_acc = np.mean([running_vl_acc_dict[x] for x in domain_list])
            # Store overall data
            running_vl_loss_dict["overall"] = vl_loss
            running_vl_acc_dict["overall"] = vl_acc

            # WANDB Logs
            wandb_loss_log = {}
            for key, val in running_vl_loss_dict.items():
                wandb_loss_log[key + "_vl_loss"] = val
            wandb_acc_log = {}
            for key, val in running_vl_acc_dict.items():
                wandb_acc_log[key + "_vl_acc"] = val
            wandb.log(wandb_loss_log, step=self.iteration)
            wandb.log(wandb_acc_log, step=self.iteration)
            # On-screen performance
            print("-----------------------------------")
            print("Fine-grained validation performance")
            print("-----------------------------------")
            print("Loss")
            pprint(running_vl_loss_dict)
            print("Accuracy")
            pprint(running_vl_acc_dict)
            print("-----------------------------------")

            # Performance on target domains
            if self.target_domains is not None:
                for domain in self.target_domains:
                    self.test_split_obj.set_domain_spec_mode(True, domain)
                    temp_loss, temp_acc = self.validate("test")
                    running_ts_loss_dict[domain] = temp_loss
                    running_ts_acc_dict[domain] = temp_acc.data.item()
                    self.test_split_obj.set_domain_spec_mode(False)

                # Calculate overall performance
                ts_loss = np.mean(
                    [running_ts_loss_dict[x] for x in self.target_domains]
                )
                ts_acc = np.mean([running_ts_acc_dict[x] for x in self.target_domains])
                running_ts_loss_dict["overall"] = ts_loss
                running_ts_acc_dict["overall"] = ts_acc

                # WANDB Logs
                # Log losses of all domains in a single plot
                wandb_loss_log = {}
                for key, val in running_ts_loss_dict.items():
                    wandb_loss_log[key + "_ts_loss"] = val
                wandb_acc_log = {}
                for key, val in running_ts_acc_dict.items():
                    wandb_acc_log[key + "_ts_acc"] = val
                wandb.log(wandb_loss_log, step=self.iteration)
                wandb.log(wandb_acc_log, step=self.iteration)

                # Log accuracies of all domains in a single plot
                # On-screen performance
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
                    self.ckpt_folder + "/model_ep_" + str(self.epoch) + ".pth",
                    self.params,
                )

            # Check for epoch level scheduler step
            if self.scheduler is not None:
                if self.scheduler_mode == "epoch":
                    if self.params.OPTIM.LEARNING_RATE_SCHEDULER == "exp":
                        self.scheduler.step()
                    elif self.params.OPTIM.LEARNING_RATE_SCHEDULER == "invlr":
                        self.scheduler.step()

            score = vl_acc
            if best_score is None or score > best_score:
                best_score = score
                self.saveModel(self.ckpt_folder + "/best_so_far.pth", self.params)
                with open(self.ckpt_folder + "/val_loss.json", "w") as f:
                    json.dump(running_vl_loss_dict, f)
                with open(self.ckpt_folder + "/val_acc.json", "w") as f:
                    json.dump(running_vl_acc_dict, f)
                # Store best values in tables
                best_score_wandb_table = wandb.Table(
                    columns=["Domain", "Validation Loss", "Validation Accuracy"]
                )
                for key, val in running_vl_loss_dict.items():
                    best_score_wandb_table.add_data(
                        key,
                        str(running_vl_loss_dict[key]),
                        str(running_vl_acc_dict[key]),
                    )
                wandb.log({"Best_Score_Table": best_score_wandb_table})
