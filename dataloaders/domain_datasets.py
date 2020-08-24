import os
import sys
import csv
import json
import h5py
import copy
import torch
import config

import numpy as np

from tqdm import tqdm
from PIL import Image
from pprint import pprint
from scipy.io import loadmat
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms, utils

# PACS Domains
PACS_DOM_LIST = ["art_painting", "cartoon", "photo", "sketch"]

# DomainNet Domains
DomainNet_DOM_LIST = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


class DomainDataset(Dataset):
    def __init__(
        self, dataset_name, domain, data_split_dir, phase="train", image_transform=None
    ):
        super(DomainDataset, self).__init__()

        self.dataset_name = dataset_name
        self.domain = domain
        self.data_split_dir = data_split_dir
        self.phase = phase
        self.image_transform = image_transform

        # Load the dataset
        if self.dataset_name == "PACS":
            self.dataset = {}
            self.dataset_file = h5py.File(self.domain_filter(), "r")
            temp_imgs = np.array(self.dataset_file["images"])
            temp_labels = np.array(self.dataset_file["labels"])
            temp_imgs = temp_imgs[:, :, :, ::-1]
            temp_labels = temp_labels - 1
            self.dataset["images"] = temp_imgs
            self.dataset["labels"] = temp_labels
            self.dataset_file.close()
            self.dataset_len = self.dataset["images"].shape[0]
            self.n_classes = 7
        elif self.dataset_name == "DomainNet":
            self.dataset = None
            with h5py.File(self.domain_filter(), "r") as file:
                self.dataset_len = file["images"].shape[0]
            self.n_classes = 345
        else:
            print("Dataset not supported yet")

    def domain_filter(self):
        flist = os.listdir(self.data_split_dir)
        if self.dataset_name == "PACS":
            dom_flist = [
                x for x in flist if "hdf5" in x and self.domain in x and self.phase in x
            ]
        elif self.dataset_name == "DomainNet":
            dom_flist = [
                x for x in flist if "h5" in x and self.domain in x and self.phase in x
            ]
        else:
            print("Dataset not supported yet")
        return os.path.join(self.data_split_dir, dom_flist[0])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.dataset_name == "PACS":
            img_arr = self.dataset["images"][idx]
            img_lbl = self.dataset["labels"][idx]
        elif self.dataset_name == "DomainNet":
            if self.dataset is None:
                self.dataset = h5py.File(self.domain_filter(), "r")
            img_arr = self.dataset["images"][idx]
            img_lbl = self.dataset["labels"][idx]

        img_dom = self.domain

        # Convert the image array to an image
        img = Image.fromarray(np.uint8(img_arr))

        # Apply image transformation
        if self.image_transform:
            img = self.image_transform(img)

        return (img, img_lbl, img_dom)


class Aggregate_DomainDataset:
    def __init__(
        self,
        dataset_name,
        domain_list,
        data_split_dir,
        phase="train",
        image_transform=None,
        batch_size=64,
        num_workers=4,
        use_gpu=True,
        shuffle=True,
    ):
        super(Aggregate_DomainDataset, self).__init__()
        self.dataset_name = dataset_name
        self.domain_list = domain_list
        self.data_split_dir = data_split_dir
        self.phase = phase
        self.image_transform = image_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.shuffle = shuffle

        # Individual Data Splits
        self.indiv_datasets = {}
        for domain in self.domain_list:
            self.indiv_datasets[domain] = DomainDataset(
                self.dataset_name,
                domain,
                self.data_split_dir,
                self.phase,
                self.image_transform,
            )

        # Aggregate Data-split
        self.aggregate_dataset = ConcatDataset(list(self.indiv_datasets.values()))

        # Store the list of labels and domains
        self.instance_labels = []
        self.instance_dom = []

        for domain in self.domain_list:
            self.instance_dom += [domain] * len(self.indiv_datasets[domain])
        self.instance_dom = np.array(self.instance_dom)

        # Creating dataloaders
        self.cuda = self.use_gpu and torch.cuda.is_available()
        kwargs = (
            {"num_workers": self.num_workers, "pin_memory": True} if self.cuda else {}
        )

        self.indiv_dataloaders = {}
        for domain in self.domain_list:
            self.indiv_dataloaders[domain] = torch.utils.data.DataLoader(
                self.indiv_datasets[domain],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                **kwargs,
            )

        self.aggregate_dataloader = torch.utils.data.DataLoader(
            self.aggregate_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs,
        )

        # Set current dataloader
        self.curr_split = self.aggregate_dataset
        self.curr_loader = self.aggregate_dataloader

    def set_domain_spec_mode(self, val=True, domain=None):
        if val:
            self.curr_loader = self.indiv_dataloaders[domain]
            self.curr_split = self.indiv_datasets[domain]
        else:
            self.curr_loader = self.aggregate_dataloader
            self.curr_split = self.aggregate_dataset

    @classmethod
    def from_config(cls, config, domain_list, phase, image_transform, shuffle):
        _C = config
        return cls(
            dataset=_C.DATA.DATASET,
            domain_list=domain_list,
            data_split_dir=_C.DATA.DATA_DIR,
            phase=phase,
            image_transform=image_transform,
            batch_size=_C.DATALOADER.BATCH_SIZE,
            num_workers=_C.PROCESS.NUM_WORKERS,
            use_gpu=_C.PROCESS.USE_GPU,
            shuffle=shuffle,
        )
