import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.compose import ColumnTransformer
from typing import Any, Callable

class PandasDataset(Dataset) :
    
    def __init__(self, df : pd.DataFrame, feature_cols : list[str], target_cols : list[str], 
                 x_type : torch.dtype, y_type : torch.dtype) :
        
        if isinstance(target_cols, str) :
            target_cols = [target_cols]
            
        if isinstance(feature_cols, str) :
            feature_cols = [feature_cols]
        
        self.data = torch.tensor(df[feature_cols].values, dtype = x_type)
        self.target = torch.tensor(df[target_cols].values, dtype = y_type)
    
    def __len__(self) -> int :
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.data[idx]
        y = self.target[idx]
        
        return (x,y)
    
def pd2torch(df : pd.DataFrame, dtype = torch.float32) -> torch.Tensor :
    
    return np2torch(df.values, dtype = dtype)

def np2torch(array : np.ndarray, dtype = torch.float32) -> torch.Tensor :
    
    converted = torch.tensor(array, dtype = dtype)
    
    # Dimension compatibility - converts to scalar array for 2D classification
    if len(converted.shape) > 1 and converted.shape[1] == 1 :
        converted = converted.squeeze()
    
    return converted

def grad2vector(params) -> torch.Tensor :
    # Params must be iterable - a list or iterator

    grad_vector = parameters_to_vector([p.grad if p.grad is not None 
                                        else torch.zeros_like(p) for p in params])
    
    return grad_vector

def pd_data_transformer(transform_list : list[tuple[list[str], Any]]) -> ColumnTransformer :

    """Admits a list of tuples, where each tuple represents a list of dataframe column names to be transformed 
    by the corresponding scaler. Dataframe columns not specified will remain in the dataframe untouched.
    

    Returns:
        The desired transformer.
    """
    
    transformers = []
    
    for columns, chosen_scaler in transform_list :
        transformers.append((f"{chosen_scaler.__class__.__name__}", chosen_scaler, columns))
    
    return ColumnTransformer(transformers, remainder = "passthrough")

###### METRICS ####

def torch_grad_var(gradient_matrix, tl = None, tp = None, dim: int = 0,) -> torch.Tensor :
    
    return torch.var(gradient_matrix, dim = dim)

def torch_E_log_fprime(gradient_matrix, tl = None, tp = None, dim : int = 0,) -> torch.Tensor :
    
    # Avoid negative badness
    safe = torch.abs(gradient_matrix) + 1e-10
    
    logged = torch.log(safe)
    
    return torch.mean(logged, dim = dim)



def index_name(i : int = 0) -> str | int :
    
    mapping = {
        0 : "param",
        1 : "epoch",
    }
    
    return mapping.get(i, i)

def name_index(name : str) -> int :
    
    mapping = {
        "params" : 0,
        "epochs" : 1,
    }
    
    return mapping.get(name, 2)
