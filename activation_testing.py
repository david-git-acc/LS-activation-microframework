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
from typing import Any, Callable, Type
from visualisation import *
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


def experiment_from_df(df_train : pd.DataFrame, df_test : pd.DataFrame, model : nn.Module,
                       target_columns : str | list[str], loss = nn.CrossEntropyLoss(),
                       transform_list_x : list[tuple[list[str], Any]] = [],
                       transform_list_y : list[tuple[list[str], Any]] = [],
    dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long)) -> tuple[torch.Tensor, torch.Tensor, np.ndarray] :
    
    """
    Same as experiment() but taken directly from the dataframe to minimise boilerplate code.
    """
    
    X_transformer = pd_data_transformer(transform_list_x)
    Y_transformer = pd_data_transformer(transform_list_y)  
    
    df_train_X = df_train.drop(columns = target_columns)
    df_test_X = df_test.drop(columns = target_columns)
    df_train_Y = df_train[target_columns]
    df_test_Y = df_test[target_columns]
    
    X_type, Y_type = dtypes
    
    X_train, X_test = dfs2train_test(df_train_X, df_test_X, X_transformer, dtype = X_type)
    Y_train, Y_test = dfs2train_test(df_train_Y, df_test_Y, Y_transformer, dtype = Y_type)
    
    return experiment(X_train, X_test, Y_train, Y_test, model, my_loss = loss)
    
    

        
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
        
        X_train_kf, X_test_kf = dfs2train_test(
                           X_train.iloc[train_index],
                           X_train.iloc[test_index], 
                           X_transformer, 
                           dtype = torch.float32 )
        
        Y_train_kf, Y_test_kf = dfs2train_test(
                            Y_train.iloc[train_index],
                            Y_train.iloc[test_index],
                            Y_transformer,
                            dtype = torch.long)
        
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
    
    if size_diff > fold_sizediff_alarm_threshold :
        print(f"Warning: difference between min-and-max sizes of stratified K-folds exceeds threshold of {fold_sizediff_alarm_threshold}")
    
    # Get into the right dimension, need kfold dim as the last one (dim=2)
    kfold_tps = np.stack(kfold_tps, axis = 2)
    gradient_matrices = torch.stack(gradient_matrices, dim = 2) # Moves k to the end
    
    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return gradient_matrices, kfold_loss, kfold_tps

def post_experiment_test_grad(gms : torch.Tensor, over : str = "epochs",
                          test_suite : list[Callable] = [], test_columns : list[str] = [], 
                          kfold_aggfuncs : Callable | list[Callable] = torch.mean, kfold_columns : str | list[str] = "mean", 
                          to_csv : bool = False) -> pd.DataFrame :
    
    """
    Perform the function test suite on a designated set of test functions with k-folds, then collapses over the k-folds using
    an aggregation function (typically mean) and returns results as a Pandas dataframe.
    
    Params:
        gms : list of gradient matrices over folds (3D: (epochs, parameters, folds))
        over: dimension to check over. Either over "epochs" (dim=1) or the full "params" (dim=0). Can be set to a different axis manually.
        test_suite: list of functions to test on.
        test_columns: names of each test. If no value given, uses the function names.
        kfold_aggfunc: the aggregation function to collapse a dimension over. Always becomes mean() if number of folds = 1.

    Note: if you want to

    Returns:
        result_df: Pandas dataframe containing results.
    """
    
    if not isinstance(kfold_aggfuncs, list) : kfold_aggfuncs = [kfold_aggfuncs]
    if not isinstance(kfold_columns, list) : kfold_columns = [kfold_columns]
    
    # The dimension to marginalise is always the opposite to the independent, hence 0 -> 1, 1 -> 0
    dim = 1 - name2index(over)
    
    is_test_data = len(gms.size()) == 2
    
    # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
    if is_test_data : gms = gms[..., None]

    test_columns = validate_activation_df_column_names(test_suite, test_columns)
    kfold_columns = validate_activation_df_column_names(kfold_aggfuncs, kfold_columns)
    
    # Store everything we collect here
    test_results = []
    
    # Store the column names in a given format so easier to store
    df_columns = []
    
    for i, test_func in enumerate( test_suite ) :
        
        result = test_func(gms, dim = dim).unsqueeze(dim)
        
        for j, kfold_aggfunc in enumerate( kfold_aggfuncs ) :
        
            collapsed_result = kfold_aggfunc(result, dim = 2 )
            data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
            test_results.append(data)
            
            df_column_name = bencode_name_pair(test_columns[i], kfold_columns[j])
            df_columns.append(df_column_name)
    
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    
    result_df = pd.DataFrame(test_results, columns = df_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = "param" if dim == 0 else "epoch"
        
    return result_df    


# def post_experiment_test_testloss(tl : torch.Tensor, )

def complete_activation_loop(df_train : pd.DataFrame, df_test : pd.DataFrame,
                                   network_type : Type[nn.Module],
                                   target_columns : str | list[str],
                                   activations : list[Callable] = [nn.Tanh, nn.ReLU, LIPLo],
                                   transform_list_x : list[tuple[list[str], Any]] = [],
                                   transform_list_y : list[tuple[list[str], Any]] = [],
                                   test_suite : list[Callable] = [torch.var, torch_E_log_fprime, torch.mean],
                                   kfold_aggfuncs : list[Callable] = [torch.mean, torch.var],
                                   save_fig_folder : str = "saved_figures", save_csv_folder : str = "saved_csvs") :

    # Placate the linter
    if isinstance(target_columns, str) : target_columns = [target_columns]

    activation_names = [activation.__class__.__name__ for activation in activations]
    n_features = len(df_train.columns) - len(target_columns)
    n_classes = max(2, len( df_train[target_columns].value_counts()))

    for i, eval_type in enumerate(["train", "test"]) :
        for j, over in enumerate( ["epochs", "params"] ) :
            
            total_activation_df = []

            for k, activation in enumerate(activations) : 

                network = network_type(activation, n_inputs = n_features, n_outputs = n_classes)

                activation_name = activation_names[k]

                if eval_type == "train" : 
                    gms, _, _ = pd_strat_kfold_crossval(df_train, target_columns, 
                                                                network,
                                                                desired_X_transforms = transform_list_x,
                                                                desired_Y_transforms = transform_list_y)
                else :
                    gms, _, _ = experiment_from_df(df_train, df_test, network, target_columns, 
                                                    transform_list_x = transform_list_x,
                                                    transform_list_y = transform_list_y)
                                            
                results_df = post_experiment_test_grad(gms, over = over, test_suite = test_suite,
                    kfold_aggfuncs = kfold_aggfuncs if eval_type == "train" else torch.mean) # No point doing aggfuncs on test data
                                    
                

                results_df["activation"] = activation_name
                
                total_activation_df.append(results_df)
            
            total_activation_df = pd.concat(total_activation_df)
            
            total_activation_df.to_csv(f"{save_csv_folder}/measure_{over}_activation_{eval_type}.csv")
        
            plot_activation_data(total_activation_df, figsize_px = (1920, 1080),
                                max_samples = 50, markersize = 4,
                                savename = f"{save_fig_folder}/measure_{over}_activation_{eval_type}.png",
                                title = generate_plot_title(activation_names, 1 if eval_type == "test" else config.kfold_k, 1),
                                kde = i)