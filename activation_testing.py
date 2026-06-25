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
from dataclasses import replace
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
    
    gradient_matrix = torch.nan * torch.ones(size = (epochs, len(nabla)))
    test_loss = torch.nan * torch.ones(epochs)
    test_predictions = torch.nan * torch.ones(size = (epochs, len(Y_test_tensor)))

    for i in range(epochs) :
        my_model.train()
        
        # Stop accumulating gradient
        nabla = torch.zeros_like(nabla)
        
        for X_train_batch, Y_train_batch in training_dataloader :
            
            optim.zero_grad()
            predictions = my_model(X_train_batch)
            loss = my_loss(predictions, Y_train_batch)
            loss.backward()
            
            # Average nabla
            current_nabla = grad2vector(my_model.parameters()).detach()
            nabla += current_nabla
        
            optim.step()
            
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
                       feature_transforms : tuple[tuple[list[str], Any], ...] = (),
                       label_transforms : tuple[tuple[list[str], Any], ...] = (),
    dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long), 
    epochs : int = 500, batch_size : int = -1 ) -> experimentResult :
    
    """
    Same as experiment() but taken directly from the dataframe to minimise boilerplate code.
    """
    
    X_transformer = pd_data_transformer(feature_transforms)
    Y_transformer = pd_data_transformer(label_transforms)  
    
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
                            feature_transforms : tuple[tuple[list[str], Any], ...] = (), 
                            label_transforms : tuple[tuple[list[str], Any], ...] = (),
                            batch_size : int = -1) -> experimentResult :
    
    skf = StratifiedKFold(n_splits = k, random_state = config.seed, shuffle = True)
    
    # If we don't reload then it becomes useless
    original_network_state = copy.deepcopy(network.state_dict())
    
    gradient_matrices = [] # Store as list because we don't know parameters size yet
    kfold_loss = torch.ones(size = (epochs, k))
    kfold_tps = torch.nan * torch.ones(size = (epochs, len(df), k)) 
    # Kfolds have slightly different sizes - standardise to same n_samples size, then 
    # initialise all as NaN so we can filter them out later in metric calculation

    X_train = df.drop(columns = labels)
    Y_train = df[labels] 
    
    for fold_i, (train_index, test_index) in enumerate( skf.split(X_train, Y_train) ) :
        
        r = experiment_from_df(df.iloc[train_index], df.iloc[test_index], network, labels, 
                               feature_transforms = feature_transforms, 
                               label_transforms = label_transforms, 
                               dtypes = (torch.float32, torch.long),
                               epochs = epochs, batch_size = batch_size) 
        
        network.load_state_dict(original_network_state)
        
        gradient_matrices.append(r.grad)
        kfold_loss[:, fold_i] = r.testloss
        kfold_tps[:, test_index, fold_i] = r.testpreds
    
    gradient_matrices = torch.stack(gradient_matrices, dim = 2) # Moves k to the end
    
    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return experimentResult(gradient_matrices, kfold_loss, kfold_tps)

  
  
def post_experiment_test_grad(gms : torch.Tensor, over : str = "epochs",
                          test_suite : tuple[Callable, ...] = (), test_columns : list[str] = [], 
                          kfold_aggfuncs : tuple[Callable, ...] = (arithmetic_mean,), 
                          kfold_columns : list[str] = ["mean"],
                          expected_ndims : int = 2) -> pd.DataFrame :
    
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
    mp.validate(expected_ndims = expected_ndims)
     
    # The dimensions to marginalise in are always all dimensions except the dimension we care about 
    # + the kfold dimension (last)
    dim = tuple(i for i in range(len(mp.X.size()) - 1 ) if i != name2index(over) ) 
    
    # Store everything we collect here
    test_results = []
    
    # Store the column names in a given format so easier to store
    df_columns = []
    
    for i, test_func in enumerate( test_suite ) :
        
        result = test_func(mp.X, dim = dim)
        
        for j, kfold_aggfunc in enumerate( mp.kfold_aggfuncs ) :
        
            kfold_dim = len(result.size()) - 1
        
            collapsed_result = kfold_aggfunc(result, dim = kfold_dim )
            data = collapsed_result.float().view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
            test_results.append(data)
            
            df_column_name = (mp.test_columns[i], mp.kfold_columns[j])
            df_columns.append(df_column_name)
    
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    result_df = pd.DataFrame(test_results, columns = df_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = over[:-1] # Kill the "s", we view singularly
        
    return result_df    


def post_experiment_test_testloss(tl : torch.Tensor, over : None = None,
                          test_suite : None = None, test_columns : list[str] = [], 
                          kfold_aggfuncs :  tuple[Callable] = (arithmetic_mean,), 
                          kfold_columns : list[str] = ["mean"]) -> pd.DataFrame :
    
    return post_experiment_test_grad(tl, "epochs", ( testloss_dummy, ), ["test loss"], 
                                     kfold_aggfuncs, kfold_columns, expected_ndims = 1)

def post_experiment_test_testpreds(tps : torch.Tensor, over : str = "test_samples",
                          test_suite : tuple[Callable, ...] = (), test_columns : list[str] = [], 
                          kfold_aggfuncs : tuple[Callable, ...] = (arithmetic_mean,), 
                          kfold_columns : list[str] = ["mean"]) -> pd.DataFrame :
        
    return post_experiment_test_grad(tps, over, test_suite, test_columns, kfold_aggfuncs, kfold_columns)



def complete_activation_test(exp_params : experimentParams, 
                             categories : tuple[str, ...] = ()) -> dict[tuple[str,str,str], pd.DataFrame]:
    
    category_selection = (categoryParams("grad", post_experiment_test_grad, ("epochs", "params") ),
                          categoryParams("testloss", post_experiment_test_testloss, ("epochs", ) ),
                          categoryParams("testpreds", post_experiment_test_testpreds, ("epochs", "test_samples") ) )
    
    if not categories : categories = tuple( cat_param.name for cat_param in category_selection )
    
    category_params = {param.name : param for param in category_selection if param.name in set(categories)}

    n_features, n_classes = get_number_of_features_and_classes(exp_params.df_train, exp_params.labels)

    total_activation_dfs = {(eval_type, category.name, measure_type) : [] 
                            for eval_type in ("train", "test")
                            for category in category_params.values()
                            for measure_type in category.measure_types }

    for activation_index, activation in enumerate( exp_params.activations ) : 

        network = exp_params.network_type(activation, n_inputs = n_features, n_outputs = n_classes)
        
        activation_name =  exp_params.activation_names[activation_index] 
 
        r = {"train" : pd_strat_kfold_crossval(exp_params.df_train, exp_params.labels, network,
                                                feature_transforms = exp_params.feature_transforms,
                                                label_transforms = exp_params.label_transforms),
             "test" : experiment_from_df(exp_params.df_train, exp_params.df_test, network, exp_params.labels, 
                                            feature_transforms = exp_params.feature_transforms,
                                            label_transforms = exp_params.label_transforms)}
                
        for eval_type, category, measure_type in total_activation_dfs :
            
            # No point aggregating over a single fold if it's test data; 0 variance    
            aggfuncs = exp_params.kfold_aggfuncs if eval_type == "train" else ( arithmetic_mean, )         
            
            c = category_params[category]
            data = getattr(r[eval_type], category)

            results_df = c.tester(data, over = measure_type, test_suite = exp_params.test_suite, kfold_aggfuncs = aggfuncs) 
            results_df["activation"] = activation_name
            
            total_activation_dfs[(eval_type, category, measure_type) ].append(results_df)
          
    total_activation_dfs = {df_type : pd.concat(df) for df_type, df in total_activation_dfs.items()}
    
    return total_activation_dfs



def LS_alpha_sensitivity_test(exp_params : experimentParams, n_alphas : int = 5, 
                              categories : tuple[str,...] = ("testloss", )) -> dict[tuple[str, str, str], pd.DataFrame]:
    
    if len(exp_params.activations) != 1 :
        print(f"Got more than 1 activation for sensitivity test; using first choice. Please select only one.")
    
    base_activation = exp_params.activations[0] 
    
    alphas = np.linspace(0, 1, n_alphas)
    activations = []
    activation_names = []
    
    for alpha in alphas : 
        
        activation = LS(base_activation, alpha, learnable =  False)     
        activation_name = f"$\\alpha ={alpha:.2f}$"
        
        activations.append(activation)
        activation_names.append(activation_name)
    
    new_experiment_params = replace(exp_params, 
                                    activations = activations,
                                    activation_names = activation_names)
    
    return complete_activation_test(new_experiment_params, categories = categories)