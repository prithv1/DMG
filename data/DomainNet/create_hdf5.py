import os
import cv2
import sys
import json
import h5py
import random

import numpy as np

from tqdm import tqdm
from pprint import pprint
from scipy.stats import entropy

# DomainNet -- all the 6 domains
all_domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

# Path to data containing all the files
DATA_SPLIT_DIR = "../DomainNet/"


def process_txt(txt_file):
    with open(txt_file, "r") as f:
        data = [x.strip("\n") for x in f.readlines()]
    data = [x.split(" ") for x in data]
    data = [[x[0], int(x[1])] for x in data]
    return data


def create_dataset(split_txt_file, save_path):
    # Get split data
    print("Loading data from txt file..")
    split_data = process_txt(split_txt_file)

    # Get number of images
    num_instances = len(split_data)

    # Define data-shape
    im_shape = (num_instances, 224, 224, 3)
    lbl_shape = (num_instances,)

    # Open an h5 file and create earrays
    print("Opening h5 file and creating e-arrays..")
    hdf5_file = h5py.File(save_path, mode="w")
    hdf5_file.create_dataset("images", im_shape)
    hdf5_file.create_dataset("labels", lbl_shape)

    # Store labels in dataset
    print("Adding labels..")
    hdf5_file["labels"][...] = np.array([x[1] for x in split_data])

    # Store images in dataset
    print("Adding images..")
    for i in tqdm(range(num_instances)):
        curr_img_path = split_data[i][0]
        img = cv2.imread(curr_img_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save the image file
        hdf5_file["images"][i, ...] = img[None]

    hdf5_file.close()


if __name__ == "__main__":
    # Save Path prefix for all files
    path_prefix = DATA_SPLIT_DIR + "tv_0.9_splits/"

    # Create hdf5 files for all splits of all domains
    for domain in all_domains:
        print("*" * 40)
        print("*" * 40)
        print("Creating files for domain ", domain)
        train_file = path_prefix + domain + "_train.txt"
        val_file = path_prefix + domain + "_val.txt"
        test_file = path_prefix + domain + "_test.txt"

        print("Processing split train")
        create_dataset(train_file, path_prefix + domain + "_train.h5")

        print("Processing split val")
        create_dataset(val_file, path_prefix + domain + "_val.h5")

        print("Processing split test")
        create_dataset(test_file, path_prefix + domain + "_test.h5")
        print("*" * 40)
        print("*" * 40)
