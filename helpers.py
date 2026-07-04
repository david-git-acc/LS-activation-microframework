import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.compose import ColumnTransformer
from typing import Any, Callable
import hashlib
import json
from pathlib import Path
import inspect
from dataclasses import asdict
from matplotlib import pyplot as plt
import yaml
import random
from sklearn.model_selection import train_test_split


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






### REST


def get_name(obj : Any) -> str :
    
    if hasattr(obj, "__name__") :
        return obj.__name__
    
    return obj.__class__.__name__

def np2torch(array : np.ndarray, dtype = torch.float32) -> torch.Tensor :
    
    converted = torch.tensor(array, dtype = dtype)
    
    # Dimension compatibility - converts to scalar array for 2D classification
    if len(converted.shape) > 1 and converted.shape[1] == 1 :
        converted = converted.squeeze()
    
    return converted

def pd2torch(df : pd.DataFrame, dtype = torch.float32) -> torch.Tensor :
    
    return np2torch(df.values, dtype = dtype)

class PandasDataset(Dataset) :
    
    def __init__(self, df : pd.DataFrame, feature_cols : list[str], target_cols : list[str], 
                 x_type : torch.dtype, y_type : torch.dtype) :
        
        if isinstance(target_cols, str) :
            target_cols = [target_cols]
            
        if isinstance(feature_cols, str) :
            feature_cols = [feature_cols]
        
        self.data = pd2torch(df[feature_cols], dtype = x_type)
        self.target = pd2torch(df[target_cols], dtype = y_type)
    
    def __len__(self) -> int :
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.data[idx]
        y = self.target[idx]
        
        return (x, y)

def params2grad_vector(params) -> torch.Tensor :
    # Params must be iterable - a list or iterator

    grad_vector = parameters_to_vector([p.grad if p.grad is not None 
                                        else torch.zeros_like(p) for p in params])
    
    return grad_vector

def activations2tensor_2d(activation_outs : list[torch.Tensor]) -> torch.Tensor :
    
    # Marginalise out the batch dimension, don't care about individual samples and never will
    refined_activation_outs = [torch.nanmean(layer_activation.detach().cpu(), dim = 0) for layer_activation in activation_outs]
    
    # Width is the MAX width, pad the rest with nans
    width = max([layer.size()[0] for layer in refined_activation_outs])
    
    # Each row is a layer of the network, each column corresponds to a neuron in that layer
    activation_subtensor = torch.full(size = (len(activation_outs), width ), fill_value = torch.nan)

    for layer_index, activation_layer in enumerate( refined_activation_outs ) :
        activation_subtensor[layer_index, 0:len(activation_layer)] = activation_layer
        
    return activation_subtensor

def activations2tensor_1d(activation_outs : list[torch.Tensor]) -> torch.Tensor :
    # Same as above but also marginalise over layers AND neurons
    refined_activation_outs = [torch.nanmean(layer_activation.detach().cpu(), dim = (0,1)).item()
                               for layer_activation in activation_outs]
    
    return torch.tensor(refined_activation_outs)


def pd_data_transformer(transform_list : tuple[tuple[list[str], Any], ...]) -> ColumnTransformer :

    """
    Admits a list of tuples, where each tuple represents a list of dataframe column names to be transformed 
    by the corresponding scaler. Dataframe columns not specified will remain in the dataframe untouched.
    

    Returns:
        The desired transformer.
    """
    
    transformers = []
    
    for columns, chosen_scaler in transform_list :
        transformers.append((f"{get_name(chosen_scaler)}", chosen_scaler, columns))
    
    return ColumnTransformer(transformers, remainder = "passthrough")

# Done to avoid code repetition. Need to use np.asarray() to avoid linter crying 
# and .to_frame() to keep OrdinalEncoder() happy - doesn't like series objects
def compatible_torch_transform(transform : Callable, x : pd.Series | pd.DataFrame, dtype= torch.float32 ) -> torch.Tensor :
    
    x_safe = x.to_frame() if isinstance(x, pd.Series) else x
    
    transformation = transform(x_safe)
    
    converted = np2torch(np.asarray(transformation), dtype = dtype)
    
    return converted

def dfs2train_test(df_train : pd.DataFrame, df_test : pd.DataFrame, transformer,
                       dtype : torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor] :
    
    # Boilerplate
    train = compatible_torch_transform(transformer.fit_transform, df_train, dtype = dtype)
    test = compatible_torch_transform(transformer.transform, df_test, dtype = dtype)
    
    return train, test

###### METRICS ####


def variance(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    if isinstance(dim, int) :
        dim = (dim, )
    
    mean = torch.nanmean(X, dim=dim, keepdim = True)

    squares = (X - mean)**2

    return torch.nanmean(squares, dim = dim)

def arithmetic_mean(X : torch.Tensor, dim : int = 0) -> torch.Tensor :
    
    meaned = torch.nanmean(X, dim = dim)

    return meaned    

def log_average(X : torch.Tensor, dim : int = 0,) -> torch.Tensor :
    
    # Avoid negative badness
    safe = torch.abs(X) + 1e-10
    
    logged = torch.log(safe)
    
    return torch.nanmean(logged, dim = dim)


def validate_activation_df_column_names(test_suite : list[Callable] | tuple[Callable, ...], 
                                        test_columns : list[str]) -> list[str] :
    
    col_length_diff =  len(test_suite) - len(test_columns) 
    backup_column_names = [get_name(func) for func in test_suite] # If not enough column names provided
    
    # If no column names provided at all, use the default names
    if not test_columns :
        updated_test_columns = backup_column_names
    elif col_length_diff >= 0 :
        updated_test_columns = test_columns + backup_column_names[len(test_columns):] # Add the remainder as test function names
    else : 
        raise ValueError(f"More col names provided than exist functions ({len(test_columns)} vs. {len(test_suite)})")
    
    return updated_test_columns

def name2index(name : str) -> int :
    
    mapping = {
        "epochs" : 0,
        "params" : 1,
        "test_samples" : 1,
        "folds" : 2
    }
    
    return mapping.get(name, 2)

def testloss_dummy(x : torch.Tensor, dim : int = 0) -> torch.Tensor :
    
    return x

def extract_tuple_list(arr : list[tuple[Any, ...]]) -> list[list[Any]] :
    
    if len(arr) == 0 : return []
    
    n_elements_per_tuple = max(len(ntuple) for ntuple in arr)
    tuple_buckets = [set([]) for _ in range(n_elements_per_tuple)]
    
    for ntuple in arr :
        
        for tuple_index, elem in enumerate( ntuple ) :
            tuple_buckets[tuple_index].add(elem)
    
    return [list(x) for x in tuple_buckets]    

def generate_plot_title(category : str, kfold_k : int = 0) -> str :
    
    eval_type = "train" if kfold_k > 1 else "test"
    
    fold_explanation = f", {kfold_k}-fold" if kfold_k else ""
    
    title = f"[{eval_type} data{fold_explanation}] Activation tests over {category} data"
    
    return title            

def symlog(x : pd.DataFrame | np.ndarray | pd.Series, thresh : float = 1.0) -> pd.DataFrame | np.ndarray | pd.Series : 
    
    return np.sign(x) * np.log1p(np.abs(x) / thresh)

def get_number_of_features_and_classes(df : pd.DataFrame, labels : str | list[str]) -> tuple[int, int] :
    
    n_features = len(df.columns) - len(labels)
    n_classes = max(2, len( df[labels].value_counts()))
    
    return n_features, n_classes

def generate_savefolder_pd(dfs : pd.DataFrame | dict[tuple[str,str,str] , pd.DataFrame], maxlen : int = 10) -> str :
    
    if isinstance(dfs, pd.DataFrame) :
        dfs = {("it's coming to a close", "Hyper Infrasonic Relocation Oscillator", "(I'm not going away)*3") : dfs}
    
    
    strformat = {str(k) : sorted( [ str(x) for x in v.columns ] ) for k, v in dfs.items()}
    as_str = json.dumps(strformat, sort_keys = True).encode()
    
    hash_monstrosity = hashlib.sha256(as_str).hexdigest()
    
    return "data-" + hash_monstrosity[:maxlen] 
    
def create_path(pathname : str) -> None :
    
    path = Path(pathname)
    
    path.mkdir(parents = True, exist_ok = True)
    
def determine_plot_type(eval_type, category, measure_type) -> str :
    
    if measure_type == "epochs" :
        return "curve"
    elif eval_type == "train" : 
        return "histplot"
    else :
        return "kde"

def safe_asdict(config_dataclass, func) :
    """
    Returns a dictionary of parameters that the target function 
    actually accepts, filtered from the dataclass.
    """
    
    params = asdict(config_dataclass)
    sig = inspect.signature(func)
    # Return only keys that exist as valid arguments in the function signature
    return {k: v for k, v in params.items() if k in sig.parameters}

def sampling_indices(n : int, max_samples : int) -> list[int] :
    
    max_samples = min(max_samples, n)
    
    return np.round(np.linspace(0, n - 1, max_samples)).astype(int).tolist()
    
def get_n_colours(n : int, cmap : str = "viridis" ) -> list :
    
    cmap_function = plt.get_cmap(cmap)
    
    c_range = np.linspace(0, 1, n)
    
    return cmap_function(c_range).tolist()

def dummy_idfunc(x : Any) : return x

def category2measure_types(category : str) -> tuple[str, ...] :
    
    mapping = {
        "grad" : ("epochs", "params"),
        "testloss" : ("epochs",),
        "testpreds" : ("epochs", "test_samples")
    }
    
    return mapping[category]

def safe_set_params(obj, params_dict) -> None : 
    
    if not hasattr(obj, "default_params") or not isinstance(obj.default_params, dict) :
        raise AttributeError(f"Object {obj} must have default_params dictionary attribute")
    
    for k, v in params_dict.items() :
        if k in obj.default_params : 
            setattr(obj, k, v)
        else : 
            raise ValueError(f"Attempted to assign value to non-optional attribute {k}")
        
def instantiate_default_params(obj) -> dict[str, Any] :
    
    default_params = {}
    
    for attr, val in obj.__dict__.items() : 
        
        if attr.startswith("_") :
            default_params[attr] = val
    
    return default_params

def is_hashable(val : Any) -> bool :
    return isinstance(val, (int, float, str, list, tuple, dict))

def smart_str(x : Any) -> str :
    
    mapping = {
        "str" : lambda x : str(x), 
        "list" : lambda x : ", ".join([smart_str(y) for y in x]),
        "dict" : lambda x : smart_str([smart_str(k) + " : " + smart_str(v) for k, v in sorted(x.items())]),
        "tuple" : lambda x : ", ".join([smart_str(y) for y in x]),
        "int" : lambda x : str(x),
        "float" : lambda x : f"{x:.4f}"
     }
    
    x_name = type(x).__name__
    
    mapped = mapping.get(x_name, None)
    
    if mapped is None : return x_name
    return mapped(x)

def update_config(registed_params : dict[str, Callable], config : dict[str, Any], namestring = "activations") -> None :
    namestring_names = f"{namestring[:-1]}_names"
    config[namestring_names] = config.get(namestring, []) # Get rid of the "s"
    config[namestring] = [registed_params[name.lower()] 
                         for name in config[namestring_names]]
    


# Necessary
update_config({ 
    "mean" : arithmetic_mean,
    "log_average" : log_average,
    "variance" : variance
}, 
config, "test_functions")

update_config({ 
    "mean" : arithmetic_mean,
    "variance" : variance
}, 
config, "kfold_aggfuncs")