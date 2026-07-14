import torch
import pandas as pd
import numpy as np
from typing import Any, Callable
import yaml
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score

# CUSTOM
from support.torch_reducers import arithmetic_mean, variance, log_average

# INITIALISATION

def import_config(config_saveloc : str = "config.yaml") -> dict[str, Any]:
    
    with open(config_saveloc, "r") as f :
        config = yaml.safe_load(f)

    return config

config = import_config()

df = pd.read_csv("datasets/penguins.csv", index_col = 0)
df = df[config["features"] + config["labels"]].dropna(how = "any").reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size = config["test_size"])

# Fix seeds for reproducibility
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

def update_config(registed_params : dict[str, Callable], config : dict[str, Any], namestring = "activations") -> None :
    namestring_names = f"{namestring[:-1]}_names"
    config[namestring_names] = config.get(namestring, []) # Get rid of the "s"
    config[namestring] = [registed_params[name.lower()] 
                         for name in config[namestring_names]]
    

function_registry : dict[str, Callable] = { 
    "mean" : arithmetic_mean,
    "log_average" : log_average,
    "variance" : variance,
    "accuracy" : accuracy_score, 
    "balanced_accuracy" : balanced_accuracy_score,
    "mcc" : matthews_corrcoef,
    
}
    
# Necessary
update_config(function_registry, config, "reducers")
update_config(function_registry, config, "kf_reducers")
update_config(function_registry, config, "eval_metrics")