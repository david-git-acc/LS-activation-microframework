import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random
from activations import *
from networks import *
from helpers import *
from typing import Any, Callable
import config

def experiment(X_train_tensor : torch.Tensor, X_test_tensor : torch.Tensor, 
               Y_train_tensor : torch.Tensor, Y_test_tensor : torch.Tensor, 
               my_model : nn.Module, my_loss = nn.CrossEntropyLoss(),
               epochs : int = config.epochs, lr : float = 0.001, 
               batch_size : int = -1) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]: 
    
    # Use full-batch GD if no batch size given
    batch_size = len(X_train_tensor) if batch_size == -1 else batch_size
    
    training_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    training_dataloader = DataLoader(training_dataset, batch_size, shuffle = True )
    
    optim = torch.optim.Adam(my_model.parameters(), lr=lr)

    nabla = torch.zeros_like(grad2vector(my_model.parameters()))
    
    gradient_matrix = -1 * torch.ones(size = (epochs, len(nabla)))
    test_loss = -1 * torch.ones(epochs)
    test_predictions = -1 * np.ones(shape = (epochs, len(Y_test_tensor)))

    for i in range(epochs) :
        my_model.train()
        
        # Stop accumulating gradient
        nabla = torch.zeros_like(nabla)
        
        for X_train_batch, Y_train_batch in training_dataloader :
            
            optim.zero_grad()
            predictions = my_model(X_train_batch)
            loss = my_loss(predictions, Y_train_batch)
            loss.backward()
            optim.step()
            
            # Average nabla
            current_nabla = grad2vector(my_model.parameters())
            nabla += current_nabla
        
        # Recompute nabla after optimisation
        nabla /= len(training_dataloader)
        
        my_model.eval()
        with torch.no_grad() :
            
            test_predictions_torch = my_model(X_test_tensor)
            test_predictions_numpy = test_predictions_torch.detach().cpu().numpy()
            outs = np.argmax(test_predictions_numpy, axis = 1) # maximum over columns, so we get a 1D vector of predictions
            
            gradient_matrix[i, :] = nabla
            test_loss[i] = my_loss(test_predictions_torch, Y_test_tensor).item()
            test_predictions[i, :] = outs.reshape(-1)   
            
        
    # gm = 2D (epochs, parameters), TL = 1D (epochs), TP = 2D (epochs, n_test)
    return (gradient_matrix, test_loss, test_predictions)

# Done to avoid code repetition. Need to use np.asarray() to avoid linter crying 
# and .to_frame() to keep OrdinalEncoder() happy - doesn't like series objects
def compatible_torch_transform(transform : Callable, x : pd.Series | pd.DataFrame, dtype= torch.float32 ) -> torch.Tensor :
    
    x_safe = x.to_frame() if isinstance(x, pd.Series) else x
    
    transformation = transform(x_safe)
    
    converted = np2torch(np.asarray(transformation), dtype= dtype)
    
    return converted
        
def pd_strat_kfold_crossval(df : pd.DataFrame, labels : list[str], 
                            network : nn.Module,
                            k : int = 10, epochs : int = config.epochs, 
                            desired_X_transforms : list = [], 
                            desired_Y_transforms : list = [],
                            batch_size : int = -1,
                            fold_sizediff_alarm_threshold : int = 2) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    
    skf = StratifiedKFold(n_splits = k, random_state = config.seed, shuffle = True)
    
    gradient_matrices = [] # Store as list because we don't know parameters size yet
    kfold_loss = torch.ones(size = (epochs, k))
    kfold_tps = [] # Store as list because we don't know original size

    X_train = df.drop(columns = labels)
    Y_train = df[labels]
    
    X_transformer = pd_data_transformer(desired_X_transforms)
    Y_transformer = pd_data_transformer(desired_Y_transforms)
    
    for fold_i, (train_index, test_index) in enumerate( skf.split(X_train, Y_train) ) :
        
        # Boilerplate
        X_train_kf = compatible_torch_transform(
            X_transformer.fit_transform, X_train.iloc[train_index], dtype = torch.float32)
        
        X_test_kf = compatible_torch_transform(
            X_transformer.transform, X_train.iloc[test_index], dtype = torch.float32)
        
        Y_train_kf = compatible_torch_transform(
            Y_transformer.fit_transform, Y_train.iloc[train_index], dtype = torch.long)
        
        Y_test_kf = compatible_torch_transform(
            Y_transformer.transform, Y_train.iloc[test_index], dtype = torch.long)
        
        # gradient matrices, test loss
        gm, tl, tps = experiment(X_train_kf, X_test_kf, Y_train_kf, Y_test_kf, network, epochs = epochs,
                                 batch_size = batch_size)
        
        
        gradient_matrices.append(gm)
        kfold_tps.append(tps)
        kfold_loss[:, fold_i] = tl
    
    # Sometimes we get off-by-one errors in prediction shape, so we must trim
    n_preds_per_fold = [preds.shape[-1] for preds in kfold_tps]
    high = max(n_preds_per_fold)
    low = min(n_preds_per_fold)
    size_diff = high-low
    
    kfold_tps = [tps[:, :low] for tps in kfold_tps]
    
    if size_diff >= fold_sizediff_alarm_threshold :
        print(f"Warning: difference between min-and-max sizes of stratified K-folds exceeds threshold of {fold_sizediff_alarm_threshold}")
    
    # Get into the right dimension, need kfold dim as the last one (dim=2)
    kfold_tps = np.stack(kfold_tps, axis = 2)
    gradient_matrices = torch.stack(gradient_matrices, dim = 2) # Moves k to the end
    
    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return gradient_matrices, kfold_loss, kfold_tps

def post_experiment_test(gms : torch.Tensor, test_loss : torch.Tensor, test_predictions : np.ndarray, over : str = "epochs",
                          test_suite : list[Callable] = [], test_columns : list[str] = [],
                          collapse_on : int = 2, aggfunc : Callable = torch.mean ) -> pd.DataFrame :
    
    """
    Perform the function test suite on a designated set of test functions with k-folds, then collapses over the k-folds using
    an aggregation function (typically mean) and returns results as a Pandas dataframe.
    
    Params:
        gms : list of gradient matrices over folds (3D: (epochs, parameters, folds))
        test_loss: list of test losses over folds (2D: parameters, folds)
        test_predictions: list of test predictions over folds (2D: end-of-trainings, folds)
        over: dimension to check over. Either over "epochs" (dim=1) or the full "params" (dim=0). Can be set to a different axis manually.
        test_suite: list of functions to test on.
        test_columns: names of each test. If no value given, uses the function names.
        collapse_on: the dimension to collapse over. ALWAYS use k=2 (fold index) unless you know what you're doing.
        aggfunc: the aggregation function to collapse a dimension over. Always becomes mean() if number of folds = 1.

    Returns:
        result_df: Pandas dataframe containing results.
    """
    
    dim = name_index(over)
    
    is_test_data = len(gms.size()) == 2
    
    # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
    if is_test_data : 
        gms = gms[..., None]
        test_loss = test_loss[..., None]
        test_predictions = test_predictions[..., None]
        aggfunc = torch.mean # mean(x) = x for singleton x
    
    col_length_diff =  len(test_suite) - len(test_columns) 
    backup_column_names = [func.__class__.__name__ for func in test_suite] # If not enough column names provided
    
    # If no column names provided at all, use the default names
    if not test_columns :
        test_columns = backup_column_names
    elif col_length_diff >= 0 :
        test_columns += backup_column_names[len(test_columns):] # Add the remainder as test function names
    else : 
        raise ValueError(f"More test col names provided than exist test functions ({len(test_columns)} vs. {len(test_suite)})")
        
    if collapse_on == dim :
        raise ValueError(f"Cannot collapse on measuring variable (both have dim={dim})")
    
    # Store everything we collect here
    test_results = []
    
    for test_func in test_suite :
        result = test_func(gms, test_loss, test_predictions, dim = dim).unsqueeze(dim)
        
        collapsed_result = aggfunc(result, dim = collapse_on )
        data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
        test_results.append(data)
    
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    
    result_df = pd.DataFrame(test_results, columns = test_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = index_name(dim)
        
    return result_df    
