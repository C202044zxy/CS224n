#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
import torchvision

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

# Check if CUDA (GPU) is available
print(torch.version.cuda)
print(torchvision.version.cuda)
print(torchvision.__version__)
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(torch.cuda.is_available())
print("Using device:", device)

# Example usage: move a tensor to the selected device
tensor = torch.tensor([1, 2, 3])
tensor = tensor.to(device)
