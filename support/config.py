import torch
import pandas as pd
import numpy as np
from typing import Any, Callable
import yaml
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score, r2_score
from torch import nn

### CUSTOM
from networks import ShortNetwork, DiamondNetwork, ActivationNetwork, to_batchnorm
from activations import IPLo, to_LS

# CUSTOM
from support.torch_reducers import arithmetic_mean, variance, log_average, stdeviation, norm

# INITIALISATION

function_registry : dict[str, Callable] = { 
    "mean" : arithmetic_mean,
    "log_average" : log_average,
    "norm" : norm,
    "variance" : variance,
    "stdeviation" : stdeviation,
    "accuracy" : accuracy_score, 
    "balanced_accuracy" : balanced_accuracy_score,
    "mcc" : matthews_corrcoef,
    "r2" : r2_score,
}

# Necessary. Activations must always be passed as CLASSES, not as instances
activation_registry : dict[str, type[nn.Module]] = { 
    "iplo" : IPLo,
    "tanh" : nn.Tanh,
    "relu" : nn.ReLU,
}

network_registry : dict[str, type[ActivationNetwork]] = {
    "shortnetwork" : ShortNetwork,
    "diamondnetwork" : DiamondNetwork
}


def import_config(config_saveloc : str = "config.yaml") -> dict[str, Any]:
    
    with open(config_saveloc, "r") as f :
        config = yaml.safe_load(f)

    config_names2functions(activation_registry, config, "activations")
    config_names2functions(function_registry, config, "reducers")
    config_names2functions(function_registry, config, "kf_reducers")
    config_names2functions(function_registry, config, "eval_metrics")
    
    handle_activation_mods(activation_registry, config)
    handle_network_mods(network_registry, config)

    return config

def config_names2functions(registed_params : dict[str, Callable], config : dict[str, Any], namestring = "activations") -> None :
    namestring_names = f"{namestring[:-1]}_names" # Get rid of the "s"
    
    # Defines activation_names, reducer_names, etc
    config[namestring_names] = config.get(namestring, []) 
    
    # Then replaces the original namestring with the actual functions themselves, using lowercase to avoid case-sensitivity
    config[namestring] = [registed_params[name.lower()] for name in config[namestring_names]]
    

def handle_activation_mods(activation_registry : dict[str, type[nn.Module]], config : dict[str, Any]) -> None :
    
    additional_details = config.get("activation_mods", {})
    
    modified_activations = {}
    for activation_name, activation_dict in additional_details.items() :
        
        lowercase_aname = activation_name.lower()
        activation = activation_registry[lowercase_aname]
        
        if activation_dict.get("LS", False) :
            modified_activation = to_LS(activation, **activation_dict["LS"])
            details = (f"LS-{activation_dict["LS"].get("alpha", ""):.2f}[{activation_name}]", modified_activation)
            modified_activations[activation_name] = details
    
    activation_names = config["activation_names"]
    activations = config["activations"]
    for a_index, a_name in enumerate(activation_names) :
        
        if modified_activations.get(a_name, None) is None : continue
        modified_name, modified_actfunc = modified_activations[a_name]

        activation_names[a_index] = modified_name
        activations[a_index] = modified_actfunc        
                    
def handle_network_mods(network_registry : dict[str, type[ActivationNetwork]], config : dict[str, Any]) -> None :
    
    network_modifications : dict[str, dict] = config.get("network_mods", {})
    
    for network_name, network_dict in network_modifications.items() :
        
        lowercase_nname = network_name.lower()
        network = network_registry[lowercase_nname]
        
        if network_dict.get("batchnorm", False) :
            modified_network = to_batchnorm(network)
            network_registry[network_name] = modified_network
        
    config["network_type"] = network_registry[config["network_type"].lower()]        
        

config = import_config()

df = pd.read_csv("datasets/extended_flower_morphometrics.csv").head(5000)
df = df[config["features"] + config["labels"]].dropna(how = "any").reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size = config["test_size"])

# Fix seeds for reproducibility
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])


