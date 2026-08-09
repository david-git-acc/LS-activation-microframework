from typing import Any, Callable
from dataclasses import asdict
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def get_name(obj : Any) -> str :
    
    if hasattr(obj, "__name__") :
        return obj.__name__
    
    return obj.__class__.__name__


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


def extract_tuple_list(arr : list[tuple[Any, ...]]) -> list[list[Any]] :
    
    if len(arr) == 0 : return []
    
    n_elements_per_tuple = max(len(ntuple) for ntuple in arr)
    tuple_buckets = [set([]) for _ in range(n_elements_per_tuple)]
    
    for ntuple in arr :
        
        for tuple_index, elem in enumerate( ntuple ) :
            tuple_buckets[tuple_index].add(elem)
    
    return [list(x) for x in tuple_buckets]   
      


def create_path(pathname : str) -> None :
    
    path = Path(pathname)
    
    path.mkdir(parents = True, exist_ok = True)
    

def safe_asdict(config_dataclass, func : Callable) :
    """
    Returns a dictionary of parameters that the target function 
    actually accepts, filtered from the dataclass.
    """
    
    params = asdict(config_dataclass)
    sig = inspect.signature(func)
    # Return only keys that exist as valid arguments in the function signature
    return {k: v for k, v in params.items() if k in sig.parameters}


def safe_dict2params(params : dict[str, Any], func : Callable) -> dict[str, Any] :
    
    sig = inspect.signature(func)
    
    return {k : v for k, v in params.items() if k in set(sig.parameters)}

def singularise(name : str) -> str :
    
    if name.endswith("es") :
        return name[:-2] 
    
    elif name.endswith("s") and name[-2] not in {"a", "e", "i", "o", "u"} :
        return name[:-1]
    
    return name
