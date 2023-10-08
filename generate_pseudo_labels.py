#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import (CityscapesDataset, CrossCityDataset, get_test_transforms,
                      get_train_transforms)
from datasets import CrossCityDataset, get_val_transforms
from utils import ScoreUpdater, colorize_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
osp = os.path

