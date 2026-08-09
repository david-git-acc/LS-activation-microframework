import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from typing import Any, Callable
from hashlib import sha256

### CUSTOM
from support.parsing_helpers import get_name

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

def pd_data_transformer(transform_list : tuple[tuple[list[str], Callable], ...]) -> ColumnTransformer :

    """
    Admits a list of tuples, where each tuple represents a list of dataframe column names to be transformed 
    by the corresponding scaler. Dataframe columns not specified will remain in the dataframe untouched.
    

    Returns:
        The desired transformer.
    """

    transformers = []
    
    for columns, chosen_scaler in transform_list :
        transformers.append((get_name(chosen_scaler), chosen_scaler(), columns))
    
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

def symlog(x : np.ndarray, thresh : float = 1.0) -> np.ndarray : 
    
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
        (deprecated warning due to this function only being called once in the main pipeline)

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

def df2transform_list(df : pd.DataFrame) -> tuple[tuple[list[str], Callable], ...] :

    cols2transformers = (
    ( df.select_dtypes(include = "number").columns.tolist(), # Numeric columns
     StandardScaler ), 
    ( df.select_dtypes(exclude = "number").columns.tolist(), # Categorical columns
     lambda : OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1 ) ), 
    )
    
    return tuple((col, transformer) for col, transformer in cols2transformers if len(col))

    
def linreg(X : np.ndarray, y : np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]] :

    """Linear regression function on arbitrary dimensions
    given data matrix of expected shape (n_samples, n_features) and observation vector of shape (n_samples, ). 
    
    Params:
        X: Input feature matrix.
        y: observation vector / labels.

    Returns:
        np.ndarray: the weight coefficients of the linear regression of expected shape (n_features + 1, )
        Callable: the function itself, taking a np.ndarray as input and outputting a 1D flat numpy array.
    """

    if X.ndim == 1 : # For scalar arrays
        X = X.reshape(-1, 1)
    
    constant_vector = np.ones((X.shape[0], 1 ))
    A = np.hstack([X, constant_vector])

    coeffs, _, _, _ = np.linalg.lstsq(A, b = y.ravel(), rcond = None)

    w = coeffs[:-1].reshape(-1, 1)
    b = coeffs[-1]

    def reg_hyperplane_function(X_test : np.ndarray) -> np.ndarray :
        
        if X_test.ndim == 1 : # If 1D, needs to be col vector or maths breaks
            X_test = X_test.reshape(-1, 1)
        
        return (X_test @ w + b).ravel() # Want coefficients as 1D for easier unpacking and use
        
    return coeffs.ravel(), reg_hyperplane_function # Same reason


def linreg_calc_2d(x : np.ndarray, y : np.ndarray) -> tuple[tuple[float, float], np.ndarray, float] :
    
    """Performs linear regression specific to 2D, generates the predictions array over the given data and 
    returns important statistics; the weight and bias, the y plot data, and the associated R^2 score.

    Exists as a helper function for visualisation.py, but use outside of this context is perfectly acceptable.

    Params:
        x: the x-dimension. Independent variable.
        y: the y-dimension. Dependent variable and the one to be predicted.

    Returns:
        tuple: contains the weight and bias term of the regressor.
        np.ndarray: the NumPy predictions array - y_hat. 
        float: the R^2 score between the predictions and the true data input y.
    """
         
    (w, b), linear_regressor = linreg(x, y)
    y_hat = linear_regressor(x)
    R2 = r2_score(y, y_hat)

    return (w, b), y_hat, R2


def linreg_calc_3d(x : np.ndarray, y : np.ndarray, z : np.ndarray, res : int = 50,
                   ) -> tuple[tuple[float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray], float] : 
    
    """Performs linear regression specific to 3D, generates the surface plot over the given data and returns
    important statistics; the two weights and the bias, the X, Y, Z surface plot matrices, and the associated R^2 score.
    
    Exists as a helper function for visualisation.py, but use outside of this context is perfectly acceptable.
    
    Params:
        x: the x-dimension. Independent variable.
        y: the y-dimension. Independent variable.
        z: the z-diemnsion. Dependent variable and the one to be predicted.
        res: number of datapoints per axis. Will use res^2 datapoints total for the surface plot. Defaults to 50.

    Returns:
        tuple: containing the weights w1 for X dimension, w2 for Y dimension and the bias term.
        tuple: containing the surface plot matrices X, Y, Z for ideal plotting.
        float: the R^2 score of the predicted Z values vs the true ones.
    """

    # X and Y are 2 different dimensions for the same samples, so linreg will expect them as (n, 2) feature matrix
    xs_and_ys = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    
    (w1, w2, b), linear_regressor = linreg(xs_and_ys, z)
    
    # Defining bounds - if we used datapoints risk of quadratic combinatorial detonation    
    xrange = np.linspace(x.min(), x.max(), res)
    yrange = np.linspace(y.min(), y.max(), res)
    
    X, Y = np.meshgrid(xrange, yrange)
    
    # Convert to 2D to fit into the prediction format, then shape back to normal after
    prediction_surface = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    z_hat = linear_regressor(prediction_surface) 
    Z = z_hat.reshape(X.shape) # Refit back to the original shape. Both X and Y are equinumerous so doesn't matter which
    R2 = r2_score(z, linear_regressor(xs_and_ys))
    
    return (w1, w2, b), (X, Y, Z), R2



def hash_df(df : pd.DataFrame, maxlen : int = 10) -> str :

    raw_str = str(df.values.astype(str).ravel().tolist())
    hash_str = sha256(raw_str.encode("utf-8")).hexdigest()[:maxlen]
    
    return hash_str