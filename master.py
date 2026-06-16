import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt
import random
from activations import *
from networks import *
from helpers import *
from config import *
from activation_testing import *

##################### CODE #####################

df = df[features + labels].dropna(how = "any").reset_index(drop=True)

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

transform_list_x = [(numeric_columns, StandardScaler())]
transform_list_y = [(nonnumerics, OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]

df_train, df_test = train_test_split(df, test_size = test_size)

df_train_transformed = df_train.copy()
df_test_transformed = df_test.copy()

test_suite = [torch_grad_var, torch_E_log_fprime]

total_activation_df = pd.DataFrame()

for i, activation in enumerate(activations) : 

    network = ShortNetwork(activation, n_inputs = len(features), n_outputs = n_classes)

    activation_name = names[i]

    gms, kfold_loss, tps = pd_strat_kfold_crossval(df_train_transformed, labels, 
                                                   network,
                                                   desired_X_transforms = transform_list_x,
                                                   desired_Y_transforms = transform_list_y)
                            
    results_df = post_experiment_test(gms, kfold_loss, tps, over = "epochs", test_suite = test_suite, 
                                       test_columns = ["grad_var", "E_log_f'"] )
    
    results_df["activation"] = activation_name
    
    total_activation_df = pd.concat([total_activation_df, results_df])
    

print(total_activation_df)