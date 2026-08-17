import torch
import pandas as pd
import numpy as np
from typing import Any, Callable
import yaml
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, matthews_corrcoef, balanced_accuracy_score, r2_score, 
                             precision_score, recall_score, f1_score)
from torch import nn

### CUSTOM
from support.parsing_helpers import singularise
from networks import ShortNetwork, DiamondNetwork, ActivationNetwork
from network_mods import to_batchnorm, to_layernorm, to_dropout, to_residual
from activations import IPLo, to_LS
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
    "precision" : lambda y_true, y_pred : precision_score(y_true, y_pred, average = "weighted", zero_division = 0), 
    "recall" : lambda y_true, y_pred : recall_score(y_true, y_pred, average = "weighted"), 
    "f1" : lambda y_true, y_pred : f1_score(y_true, y_pred, average = "weighted")
}

# Necessary. Activations must always be passed as CLASSES, not as instances to prevent state duplication
activation_registry : dict[str, type[nn.Module]] = {
    "iplo" : IPLo,
    "tanh" : nn.Tanh,
    "relu" : nn.ReLU,
}

network_registry : dict[str, type[ActivationNetwork]] = {
    "short" : ShortNetwork,
    "diamond" : DiamondNetwork
}

activation_mod_registry : dict[str, Callable] = {
    "ls" : to_LS,
}

network_mod_registry : dict[str, Callable] = {
    "batchnorm" : to_batchnorm, 
    "layernorm" : to_layernorm,
    "dropout" : to_dropout,
    "residual" : to_residual,
}

dataset_registry : dict[str, Callable[[], pd.DataFrame]] = {
    "penguins" : lambda : pd.read_csv("datasets/penguins.csv").iloc[:, 1:], # First column is redundant, drop it
    "flowers" : lambda : pd.read_csv("datasets/extended_flower_morphometrics.csv"), 
}

def import_config(config_saveloc : str = "config.yaml") -> dict[str, Any] :

    """Main function to import and define the config.yaml in its intended format. Must not be modified outside of
    config.py.
    
    Params:
        config_saveloc: filename of the config yaml file. 
        csv_filename: filename of the CSV dataset. Currently only supports .csv files.

    Returns:
        dict[str, Any]: the config file, with all necessary parameters.
    """

    with open(config_saveloc, "r") as f :
        config = yaml.safe_load(f)

    df = dataset_registry[config["dataset"]]().head(5000) # Size limit for reasonable comp speeds. No need to keep if you have more time

    if not isinstance(config["labels"], list) :
        config["labels"] = [ config["labels"] ] # Must be a list of strings for pandas consistency

    if config["features"] == "all" :
        config["features"] = [col for col in df.columns if col not in set(config["labels"])]

    # Every dataset has its own cleaning requirements. Not possible to make a one-size-fits-all cleaner so this is my fallback
    df = df[config["features"] + config["labels"]].dropna(how = "any").reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size = config["test_size"])
    
    # Master file will need references to these
    config["df"] = df
    config["df_train"] = df_train
    config["df_test"] = df_test
    
    config["base_network_type"] = network_registry[config["network_type"].lower()]    
    
    # Avoid global mutation so others can use the registries
    config["activation_registry"] = dict(activation_registry)
    config["network_registry"] = dict(network_registry)

    config_names2functions(config["activation_registry"], config, "activations")
    config_names2functions(function_registry, config, "reducers")
    config_names2functions(function_registry, config, "kf_reducers")
    config_names2functions(function_registry, config, "eval_metrics")

    handle_activation_mods(config["activation_registry"], config)
    handle_network_mods(config["network_registry"], config)

    # Fix seeds for reproducibility
    set_seed(config["seed"])

    return config


def config_names2functions(registed_params : dict[str, Callable], config : dict[str, Any], namestring = "activations") -> None :

    """Given a registry of function names to functions, a config file and name of the set of functions, map each
    function name to its corresponding function by modifying config inplace. Not to be used outside config.py.
    
    Params:
        registered_params: registry that maps function names to corresponding functions.
        config: config file in progress for modification.
        namestring: name of the set of functions, e.g activations, reducers or eval_metrics.
    """

    namestring_names = f"{singularise(namestring)}_names" # Get rid of the "s"

    # Defines activation_names, reducer_names, etc
    config[namestring_names] = config.get(namestring, [])
    
    # Then replaces the original namestring with the actual functions themselves, using lowercase to avoid case-sensitivity
    config[namestring] = [registed_params[name.lower()] for name in config[namestring_names]]

def handle_activation_mods(activation_registry : dict[str, type[nn.Module]], config : dict[str, Any]) -> None :

    """Given a registry of activation functions and config, modify config inplace to apply modifications
    to the activation functions, most notably LS. Note that config_names2functions should already have been
    performed, or else the activations will still be in string form and the code won't work. 
    
    Remember to pass in a copy of activation_registry and not the original, since it will be modified in-place.
    
    Params:
        activation_registry: the registry of activation names to their corresponding activations.
        config: the config file to be modified in-place.
    """

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
    
    """Given a registry of network string names to networks and config, modify the latter inplace to 
    apply network changes e.g BatchNorm to the networks. Please pass in a copy of network_registry, or 
    modifications will occur inplace and cause state mutation.
    
    Params:
        network_registry: the mapping of network type names to their corresponding network classes.
        config: the config file itself.
    """

    network_modifications : dict[str, dict] = config.get("network_mods", {})
    for network_name, network_dict in network_modifications.items() :

        lowercase_nname = network_name.lower()

        for network_mod_name, network_mod in network_mod_registry.items() :
            network_mod_info = network_dict.get(network_mod_name, False)
            
            if network_mod_info is True : # Some mods have no extra params; these will just be bools
                network_registry[network_name] = network_mod(network_registry[lowercase_nname])
                
            elif isinstance(network_mod_info, dict) : # Others are dicts of params, pass these in 
                network_registry[network_name] = network_mod(network_registry[lowercase_nname], **network_mod_info)  
    
    # Use lower case so user doesn't trip up on exact casing             
    config["network_type"] = network_registry[config["network_type"].lower()]        
    
def set_seed(seed : int) -> None :
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
# Technically, config impacts global state by setting default seed, but given the use-case this is acceptable
config = import_config("config.yaml") 