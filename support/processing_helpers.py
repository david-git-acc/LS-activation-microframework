import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from typing import Any, Callable

# Used as a NaN equivalent for long data
nan_long = -99999999

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
                                        else torch.zeros_like(p) for p in params]).detach().cpu()
    
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

def get_max_shape(tensors : list[torch.Tensor]) -> tuple :
    
    max_ndims = max(len(tensor.size()) for tensor in tensors)
    max_shape = [-1 for _ in range(max_ndims)]
    
    for tensor in tensors :
        shape = tensor.size()
        for dim, n_elems in enumerate( shape ) :
            max_shape[dim] = max(max_shape[dim], n_elems)
    
    return tuple(max_shape)
    

def pad_torch_stack(tensors : list[torch.Tensor], pad_with : float = torch.nan) -> list[torch.Tensor] :

    new_tensors = []
    max_shape = get_max_shape(tensors)
    max_ndims = len(max_shape)
    dtype = tensors[0].dtype # Assumes every tensor is the same datatype

    for tensor in tensors :
        assert tensor.dtype == dtype
        
        tensor_shape = tuple(tensor.size())
        ndims = len(tensor.size())
        dim_diff = max_ndims - ndims
        new_shape = tensor_shape + (1,) * dim_diff
        
        intermediate_tensor = tensor.view(*new_shape)
        
        new_tensor = torch.full(max_shape, fill_value = pad_with, dtype = dtype)
        tensor_slices = tuple(slice(0, n_elems) for n_elems in tensor_shape)
        new_tensor[tensor_slices] = intermediate_tensor
        new_tensors.append(new_tensor)
    
    return new_tensors

def pd_data_transformer(transform_list : tuple[tuple[list[str], Any], ...]) -> ColumnTransformer :

    """
    Admits a list of tuples, where each tuple represents a list of dataframe column names to be transformed 
    by the corresponding scaler. Dataframe columns not specified will remain in the dataframe untouched.
    

    Returns:
        The desired transformer.
    """
    
    transformers = []
    
    for columns, chosen_scaler in transform_list :
        scaler_name = (chosen_scaler.__name__ if hasattr(chosen_scaler, "__name__") 
                       else chosen_scaler.__class__.__name__)
        transformers.append((f"{scaler_name}", chosen_scaler, columns))
    
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

def symlog(x : pd.DataFrame | np.ndarray | pd.Series, thresh : float = 1.0) -> pd.DataFrame | np.ndarray | pd.Series : 
    
    return np.sign(x) * np.log1p(np.abs(x) / thresh)

def get_number_of_features_and_classes(df : pd.DataFrame, labels : str | list[str]) -> tuple[int, int] :
    
    n_features = len(df.columns) - len(labels)
    n_classes = max(2, len( df[labels].value_counts()))
    
    return n_features, n_classes

def dummy_idfunc(x : Any) : return x

def sampling_indices(n : int, max_samples : int) -> list[int] :
    
    max_samples = min(max_samples, n)
    
    return np.round(np.linspace(0, n - 1, max_samples)).astype(int).tolist()


def dfs_settings2tensors(df_train : pd.DataFrame, df_test : pd.DataFrame, 
      feature_transforms : tuple[tuple[list[str], Callable], ...], label_transforms : tuple[tuple[list[str], Callable], ...],
      labels : str | list[str], dtypes : tuple[torch.dtype, torch.dtype]
      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :

    """Function to convert dataframes and metadata into the correct train and test, feature and label tensors.
        MUST be deterministic. Do not use nondeterministic transformers to prevent label mismatch in metrics checking.

    Returns:
        4-tuple of tensors.
    """
    
    X_transformer = pd_data_transformer(feature_transforms)
    Y_transformer = pd_data_transformer(label_transforms)  
    
    df_train_X = df_train.drop(columns = labels)
    df_test_X = df_test.drop(columns = labels)
    df_train_Y = df_train[labels]
    df_test_Y = df_test[labels]
    
    X_type, Y_type = dtypes
    
    # Learn the transform on the training data and apply to the test data
    X_train, X_test = dfs2train_test(df_train_X, df_test_X, X_transformer, dtype = X_type)
    Y_train, Y_test = dfs2train_test(df_train_Y, df_test_Y, Y_transformer, dtype = Y_type)
    
    # Must remain in this order; other modules depend on it, and changing it will require refactors elsewhere
    return X_train, X_test, Y_train, Y_test

