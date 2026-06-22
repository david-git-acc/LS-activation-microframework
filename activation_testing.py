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
from dataclass_objects import categoryParams, experimentResult, experimentParams, monitorParams
import config
import copy

def experiment(X_train_tensor : torch.Tensor, X_test_tensor : torch.Tensor, 
               Y_train_tensor : torch.Tensor, Y_test_tensor : torch.Tensor, 
               my_model : nn.Module, my_loss = nn.CrossEntropyLoss(),
               epochs : int = config.epochs, lr : float = 0.001, 
               batch_size : int = -1) -> experimentResult : 
    
    # Use full-batch GD if no batch size given
    batch_size = len(X_train_tensor) if batch_size == -1 else batch_size
    
    training_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    training_dataloader = DataLoader(training_dataset, batch_size, shuffle = True )
    
    optim = torch.optim.Adam(my_model.parameters(), lr=lr)
    nabla = torch.zeros_like(grad2vector(my_model.parameters()))
    
    gradient_matrix = -1 * torch.ones(size = (epochs, len(nabla)))
    test_loss = -1 * torch.ones(epochs)
    test_predictions = -1 * torch.ones(size = (epochs, len(Y_test_tensor)))

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
            current_nabla = grad2vector(my_model.parameters()).detach()
            nabla += current_nabla
        
        # Recompute nabla after optimisation
        nabla /= len(training_dataloader)
        
        my_model.eval()
        with torch.no_grad() :
            
            test_predictions_torch = my_model(X_test_tensor)
            outs = torch.argmax(test_predictions_torch, dim = 1) # maximum over columns, so we get a 1D vector of predictions
            
            gradient_matrix[i, :] = nabla
            test_loss[i] = my_loss(test_predictions_torch, Y_test_tensor).item()
            test_predictions[i, :] = outs.view(-1)   
            
        
    # gm = 2D (epochs, parameters), TL = 1D (epochs), TP = 2D (epochs, n_test)
    return experimentResult(gradient_matrix, test_loss, test_predictions)


def experiment_from_df(df_train : pd.DataFrame, df_test : pd.DataFrame, model : nn.Module,
                       labels : str | list[str], loss = nn.CrossEntropyLoss(),
                       feature_transform_list : list[tuple[list[str], Any]] = [],
                       label_transform_list : list[tuple[list[str], Any]] = [],
    dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long), 
    epochs : int = 500, batch_size : int = -1 ) -> experimentResult :
    
    """
    Same as experiment() but taken directly from the dataframe to minimise boilerplate code.
    """
    
    X_transformer = pd_data_transformer(feature_transform_list)
    Y_transformer = pd_data_transformer(label_transform_list)  
    
    df_train_X = df_train.drop(columns = labels)
    df_test_X = df_test.drop(columns = labels)
    df_train_Y = df_train[labels]
    df_test_Y = df_test[labels]
    
    X_type, Y_type = dtypes
    
    X_train, X_test = dfs2train_test(df_train_X, df_test_X, X_transformer, dtype = X_type)
    Y_train, Y_test = dfs2train_test(df_train_Y, df_test_Y, Y_transformer, dtype = Y_type)
    
    return experiment(X_train, X_test, Y_train, Y_test, model, my_loss = loss, epochs = epochs, batch_size = batch_size)

        
        
def pd_strat_kfold_crossval(df : pd.DataFrame, labels : str | list[str], 
                            network : nn.Module,
                            k : int = 10, epochs : int = config.epochs, 
                            feature_transform_list : list = [], 
                            label_transform_list : list = [],
                            batch_size : int = -1,
                            fold_sizediff_alarm_threshold : int = 2) -> experimentResult :
    
    skf = StratifiedKFold(n_splits = k, random_state = config.seed, shuffle = True)
    
    # If we don't reload then it becomes useless
    original_network_state = copy.deepcopy(network.state_dict())
    
    gradient_matrices = [] # Store as list because we don't know parameters size yet
    kfold_loss = torch.ones(size = (epochs, k))
    kfold_tps = [] # Store as list because we don't know original size

    X_train = df.drop(columns = labels)
    Y_train = df[labels] 
    
    for fold_i, (train_index, test_index) in enumerate( skf.split(X_train, Y_train) ) :
        
        r = experiment_from_df(df.iloc[train_index], df.iloc[test_index], network, labels, 
                               feature_transform_list = feature_transform_list, 
                               label_transform_list = label_transform_list, 
                               dtypes = (torch.float32, torch.long),
                               epochs = epochs, batch_size = batch_size) 
        
        network.load_state_dict(original_network_state)
        
        gradient_matrices.append(r.grad)
        kfold_tps.append(r.testpreds)
        kfold_loss[:, fold_i] = r.testloss
    
    # Sometimes we get off-by-one errors in prediction shape, so we must trim
    n_preds_per_fold = [preds.size()[-1] for preds in kfold_tps]
    high = max(n_preds_per_fold)
    low = min(n_preds_per_fold)
    size_diff = high-low
    
    kfold_tps = [tps[:, :low] for tps in kfold_tps]
    
    if size_diff > fold_sizediff_alarm_threshold :
        print(f"Warning: difference between min-and-max sizes of stratified K-folds exceeds threshold of {fold_sizediff_alarm_threshold}")
    
    # Get into the right dimension, need kfold dim as the last one (dim=2)
    kfold_tps = torch.stack(kfold_tps, dim = 2)
    gradient_matrices = torch.stack(gradient_matrices, dim = 2) # Moves k to the end
    
    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return experimentResult(gradient_matrices, kfold_loss, kfold_tps)

  
  
def post_experiment_test_grad(gms : torch.Tensor, over : str = "epochs",
                          test_suite : list[Callable] = [], test_columns : list[str] = [], 
                          kfold_aggfuncs : list[Callable] = [torch.mean], 
                          kfold_columns : str | list[str] = "mean") -> pd.DataFrame :
    
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
    
    mp = monitorParams(gms, test_suite, test_columns, kfold_aggfuncs, kfold_columns)
    mp.validate()
     
    # The dimension to marginalise is always the opposite to the independent, hence 0 -> 1, 1 -> 0
    dim = 1 - name2index(over)
    
    # Store everything we collect here
    test_results = []
    
    # Store the column names in a given format so easier to store
    df_columns = []
    
    for i, test_func in enumerate( test_suite ) :
        
        result = test_func(mp.X, dim = dim).unsqueeze(dim)
        
        for j, kfold_aggfunc in enumerate( mp.kfold_aggfuncs ) :
        
            collapsed_result = kfold_aggfunc(result, dim = 2 )
            data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
            test_results.append(data)
            
            df_column_name = bencode_name_pair(mp.test_columns[i], mp.kfold_columns[j])
            df_columns.append(df_column_name)
    
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    
    result_df = pd.DataFrame(test_results, columns = df_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = over[:-1] # Kill the "s", we view singularly
        
    return result_df    


def post_experiment_test_testloss(tl : torch.Tensor, over : None = None,
                          test_suite : None = None, test_columns : list[str] = [], 
                          kfold_aggfuncs :  list[Callable] = [torch.mean], 
                          kfold_columns : str | list[str] = "mean") -> pd.DataFrame :
    
    mp = monitorParams(tl, [], [], kfold_aggfuncs, kfold_columns)
    mp.validate(expected_ndims = 1)

    # Store everything we collect here
    test_results = []
    
    # Store the column names in a given format so easier to store
    df_columns = []
    
    for j, kfold_aggfunc in enumerate( mp.kfold_aggfuncs ) :
    
        collapsed_result = kfold_aggfunc(mp.X, dim = 1 ) # 1 is the k-fold dimensino
        data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
        test_results.append(data)
        
        # Keep the encoding despite not needing it, so it's compatible with the visualisation code
        df_column_name = bencode_name_pair("test loss", mp.kfold_columns[j])
        df_columns.append(df_column_name)
    
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    
    result_df = pd.DataFrame(test_results, columns = df_columns)
    result_df.index.name = "epoch"
        
    return result_df   


def post_experiment_test_testpreds(tps : torch.Tensor, over : str = "test_samples",
                          test_suite : list[Callable] = [], test_columns : list[str] = [], 
                          kfold_aggfuncs : list[Callable] = [torch.mean], 
                          kfold_columns : str | list[str] = "mean") -> pd.DataFrame :
        
    return post_experiment_test_grad(tps, over, test_suite, test_columns, kfold_aggfuncs, kfold_columns)



def complete_activation_loop(exp_params : experimentParams, save_fig_folder : str = "saved_figures", 
                             save_csv_folder : str = "saved_csvs", categories : list[categoryParams] = []) -> None :
    
    if not categories : categories = [categoryParams("grad", post_experiment_test_grad, ["epochs", "params"]),
                                      categoryParams("testloss", post_experiment_test_testloss, ["epochs"] ),
                                      categoryParams("testpreds", post_experiment_test_testpreds, ["epochs", "test_samples"])]
    category_params = {param.name : param for param in categories}

    # Placate the linter + consistency
    if isinstance(exp_params.labels, str) : exp_params.labels = [exp_params.labels]

    n_features, n_classes = get_number_of_features_and_classes(exp_params.df_train, exp_params.labels)

    for eval_index, eval_type in enumerate(["train", "test"]) :
        
        total_activation_dfs = {(category.name, measure_type) : [] 
                                for category in category_params.values()
                                for measure_type in category.measure_types }
        
        for activation in exp_params.activations : 

            network = exp_params.network_type(activation, n_inputs = n_features, n_outputs = n_classes)
            activation_name = activation.__class__.__name__

            if eval_type == "train" : 
                r = pd_strat_kfold_crossval(exp_params.df_train, exp_params.labels, network,
                                                    feature_transform_list = exp_params.feature_transform_list,
                                                    label_transform_list = exp_params.label_transform_list)
            else :
                r = experiment_from_df(exp_params.df_train, exp_params.df_test, network, exp_params.labels, 
                                                feature_transform_list = exp_params.feature_transform_list,
                                                label_transform_list = exp_params.label_transform_list)
            
            for category, measure_type in total_activation_dfs :
                
                # No point aggregating over a single fold if it's test data; 0 variance    
                aggfuncs = exp_params.kfold_aggfuncs if eval_type == "train" else [torch.mean]         
                
                c = category_params[category]

                results_df = c.tester(getattr(r, category), over = measure_type, 
                                                    test_suite = exp_params.test_suite, kfold_aggfuncs = aggfuncs) 
                results_df["activation"] = activation_name
                
                total_activation_dfs[( category, measure_type) ].append(results_df)
            
        for category, measure_type in total_activation_dfs :              
                
            total_activation_df = pd.concat(total_activation_dfs[(category, measure_type)])
            
            savename = f"{category}-{measure_type}_on_{eval_type}"
                
            total_activation_df.to_csv(f"{save_csv_folder}/{savename}.csv")
                
            plot_activation_data(total_activation_df, figsize_px = (1920, 1080),
                            max_samples = 50, markersize = 4,
                            savename = f"{save_fig_folder}/{savename}.png",
                            title = generate_plot_title(category, 1 if eval_type == "test" else exp_params.kfold_k),
                            kde = eval_index, n_skip = 5)