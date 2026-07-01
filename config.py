import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt
import random
from activations import *
from networks import *
from helpers import *
from typing import Any, Callable

# SELECT MEASURING DEVICES HERE
activations = ( IPLo(), nn.Tanh() )
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"] # Must be numeric
labels = ["sex"] # Must be categorical

# Hyperparameters
epochs = 500
test_size = 0.2
seed = 42
lr = 0.005
sample_rate = 1
kfold_k = 10

