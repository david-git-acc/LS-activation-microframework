import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.compose import ColumnTransformer
from typing import Any, Callable
import matplotlib.pyplot as plt
from math import ceil

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

def grad2vector(params) -> torch.Tensor :
    # Params must be iterable - a list or iterator

    grad_vector = parameters_to_vector([p.grad if p.grad is not None 
                                        else torch.zeros_like(p) for p in params])
    
    return grad_vector

def pd_data_transformer(transform_list : list[tuple[list[str], Any]]) -> ColumnTransformer :

    """
    Admits a list of tuples, where each tuple represents a list of dataframe column names to be transformed 
    by the corresponding scaler. Dataframe columns not specified will remain in the dataframe untouched.
    

    Returns:
        The desired transformer.
    """
    
    transformers = []
    
    for columns, chosen_scaler in transform_list :
        transformers.append((f"{chosen_scaler.__class__.__name__}", chosen_scaler, columns))
    
    return ColumnTransformer(transformers, remainder = "passthrough")

# Done to avoid code repetition. Need to use np.asarray() to avoid linter crying 
# and .to_frame() to keep OrdinalEncoder() happy - doesn't like series objects
def compatible_torch_transform(transform : Callable, x : pd.Series | pd.DataFrame, dtype= torch.float32 ) -> torch.Tensor :
    
    x_safe = x.to_frame() if isinstance(x, pd.Series) else x
    
    transformation = transform(x_safe)
    
    converted = np2torch(np.asarray(transformation), dtype= dtype)
    
    return converted

def dfs2train_test(df_train : pd.DataFrame, df_test : pd.DataFrame, transformer,
                       dtype : torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor] :
    
    # Boilerplate
    train = compatible_torch_transform(transformer.fit_transform, df_train, dtype = dtype)
    test = compatible_torch_transform(transformer.transform, df_test, dtype = dtype)
    
    return train, test

###### METRICS ####

def torch_grad_avg(gradient_matrix, dim : int = 0) -> torch.Tensor :
    
    return torch.mean(gradient_matrix, dim = dim)

def torch_grad_var(gradient_matrix, dim : int = 0,) -> torch.Tensor :
    
    return torch.var(gradient_matrix, dim = dim)

def log_average(gradient_matrix, dim : int = 0,) -> torch.Tensor :
    
    # Avoid negative badness
    safe = torch.abs(gradient_matrix) + 1e-10
    
    logged = torch.log(safe)
    
    return torch.mean(logged, dim = dim)


def validate_activation_df_column_names(test_suite, test_columns) :
    
    col_length_diff =  len(test_suite) - len(test_columns) 
    backup_column_names = [func.__name__ for func in test_suite] # If not enough column names provided
    updated_test_columns = []
    
    # If no column names provided at all, use the default names
    if not test_columns :
        updated_test_columns = backup_column_names
    elif col_length_diff >= 0 :
        updated_test_columns = test_columns + backup_column_names[len(test_columns):] # Add the remainder as test function names
    else : 
        raise ValueError(f"More test col names provided than exist test functions ({len(test_columns)} vs. {len(test_suite)})")
    
    return updated_test_columns

def name2index(name : str) -> int :
    
    mapping = {
        "epochs" : 0,
        "params" : 1,
        "test_samples" : 1,
        "folds" : 2
    }
    
    return mapping.get(name, 2)

encoding_char_pairs = ["[]", "()", "{}", "£$", ",'"]
char_pairs = {k[1] : k[0] for k in encoding_char_pairs}

def bencode_name_pair(name_A : str, name_B : str) -> str :
    
    charset = set(list(name_A + name_B))
    
    i = 0
    for encode_key in encoding_char_pairs :
        if encode_key[0] not in charset and encode_key[1] not in charset :
            break
        i += 1
    
    if i >= len(encoding_char_pairs) :
        raise IndexError(f"Could not find viable encoding character pairs for names {name_A}, {name_B}")
    
    a, b = list(encoding_char_pairs[i])
    
    encoding = name_A + a + name_B + b
    
    return encoding

def bdecode_name_pair(encoded_name : str) -> tuple[str, str]:
    
    b = encoded_name[-1]
    a = char_pairs.get(b, None)
    
    if a is None : raise ValueError(f"Name {encoded_name} does not correspond to a pair encoding")
    
    index_of_a = encoded_name.index(a)
    
    name_A = encoded_name[:index_of_a]
    name_B = encoded_name[index_of_a+1:-1]
    
    charset = set(list(name_A + name_B))
    
    if a in charset or b in charset :
        raise ValueError(f"Name was not bracket-encoded; one of ({a},{b}) appears more than once")
    
    return name_A, name_B

def extract_bencoded_list(arr : list[str], as_lists : bool = True) -> tuple[list | set[str], list | set[str]] :
    
    first_elems = set([])
    second_elems = set([])
    
    for x in arr :
        
        A,B = bdecode_name_pair(x)
        first_elems.add(A)
        second_elems.add(B)
    
    if as_lists :
        first_elems = list(first_elems)
        second_elems = list(second_elems)
        
    return first_elems, second_elems

def validate_testfunction(X : torch.Tensor, test_suite, test_columns, kfold_aggfuncs, kfold_columns,
                          expected_ndims : int = 2) :
    
    if not isinstance(kfold_aggfuncs, list) : kfold_aggfuncs = [kfold_aggfuncs]
    if not isinstance(kfold_columns, list) : kfold_columns = [kfold_columns]
    
    is_test_data = len(X.size()) == expected_ndims
    
    # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
    if is_test_data : X = X[..., None]

    test_columns = validate_activation_df_column_names(test_suite, test_columns)
    kfold_columns = validate_activation_df_column_names(kfold_aggfuncs, kfold_columns)
    
    return X, test_suite, test_columns, kfold_aggfuncs, kfold_columns

def generate_plot_title(category : str, kfold_k : int = 0) -> str :
    
    eval_type = "train" if kfold_k > 1 else "test"
    
    fold_explanation = f", {kfold_k}-fold" if kfold_k else ""
    
    title = f"[{eval_type} data{fold_explanation}] Activation tests over symlogged {category} data"
    
    return title            

def symlog(x : float) -> float : 
    
    return np.sign(x) * np.log1p(np.abs(x))

def get_number_of_features_and_classes(df : pd.DataFrame, labels : list[str]) :
    
    n_features = len(df.columns) - len(labels)
    n_classes = max(2, len( df[labels].value_counts()))
    
    return n_features, n_classes