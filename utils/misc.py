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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.fill_(0)
