import numpy as np
import torch
from torch import nn
import pandas as pd
from dataclasses import replace
from rich.progress import Progress

# CUSTOM
from support.processing_helpers import (sampling_indices, dfs_settings2tensors, 
                                        get_number_of_features_and_classes)
from support.config import config
from support.parsing_helpers import safe_asdict
from dataclass_objects.config_objects import expConfig
from dataclass_objects.result_objects import activationResults
from categories.category_registry import category_registry
from activations import to_LS
from activation_testing.internal import multiseed_test_from_df, skf_crossval, evaluate_activation_results

def complete_activation_test(xpc : expConfig, verbose : bool = False) -> activationResults :
    
    """
    Orchestrator / god function to perform entire experiment, from training to kfold and testing given a set of 
    expConfig. Designed to minimise boilerplate and facilitate ease of use + modularity. 
    
    Params:
        xpc: the expConfig dataclass containing all important data about the experiment. 
        verbose: boolean detailing whether to provide details over current execution cycle.

    NOTE: if no category is specified in expConfig, it will use all categories. Remember to add category parameters
    to the expConfig dataclass inside dataclass_objects.py.
    
    Category selection is a tuple containing category parameters, each of which stores the triple combination of 
    category name, the associated tester function over that category (not to be confused with test functions,
    which operate directly over data (e.g mean)), and the tuple of all valid measurement types. S

    Returns:
        total_activation_dfs: a dictionary containing for each triple combination of evaluation type, category and
        measurement type, the corresponding dataframe of all associated results. 
        Each dataframe stores an agg-type test-type combination, e.g "('log_average', 'mean')" as a column name.
    """
    
    # "all" will have been converted into ("all", ) by expConfing's input validation
    categories = tuple(category_registry.keys()) if xpc.categories == ("all", ) else xpc.categories
    category_params = {cat : category_registry[cat] for cat in categories}
    
    n_features, n_classes = get_number_of_features_and_classes(xpc.df_train, xpc.labels)

    # Add to these incrementally and then concatenate at the end to turn into dataframes.
    total_activation_dfs = {(eval_type, category, measure_type) : [] 
                            for eval_type in ("train", "test")
                            for category in category_params.keys()
                            for measure_type in category_params[category].measure_types }
    
    with Progress() as progress :
        work = progress.add_task("Experiment progress:", total = len(xpc.activations) * len(total_activation_dfs))
        
        for activation_index, activation in enumerate( xpc.activations ) : 
            
            network = xpc.network_type(activation, n_inputs = n_features, n_outputs = n_classes)
            activation_name = xpc.activation_names[activation_index] 
            
            net_act_str = f"[Network: {network.name}, Activation: {activation.__name__}]"  
            if verbose : progress.console.log(f"Executing configuration: {net_act_str}")

            # Pass in params dataclass directly because function signature may become arbitrarily long with more additions
            r = {"train" : skf_crossval(**safe_asdict(xpc, skf_crossval), model = network, df = xpc.df_train),
                 "test" : multiseed_test_from_df(**safe_asdict(xpc, multiseed_test_from_df), model = network)}

            for eval_type, category, measure_type in total_activation_dfs:
                if verbose : progress.console.log(f" -> {eval_type.title()}ing on {category} data over {measure_type}")
                
                results_df = evaluate_activation_results(xpc = xpc, 
                                                     exp_result = r[eval_type], 
                                                     tester = category_params[category].tester, 
                                                     typeof_result = (eval_type, category, measure_type))
                results_df["activation"] = activation_name # So we can keep track
                
                total_activation_dfs[(eval_type, category, measure_type)].append(results_df)
                progress.advance(work, 1)

    # Only concat at the end for speed
    total_activation_dfs = {df_type : pd.concat(df, axis = 0) for df_type, df in total_activation_dfs.items()}
    
    # Wrap in the activationResults dataclass for more flexibility later
    return activationResults(total_activation_dfs)


def LS_alpha_sensitivity_test(xpc : expConfig, verbose : bool = True) -> tuple[activationResults, expConfig] :
    
    """
    Perform an alpha sensitivity test on an LS-converted activation function. Note that this implicitly assumes that
    the function specified is already in the S function family. An LS function exists in the form
    
    LS_{alpha}(f(x) E S) = alpha x + (1 - alpha) ( f(x) / f'(0) ) : alpha in (0,1)
    
    Creates as many equally spaced alphas as specified in the experiment parameters between 0 and 1 exclusive. Then,
    applies the standard complete activation test orchestrator function. Accepts the function from the 
    xpc tuple of activation functions; if there is more than one, takes the first only. 

    Params:
        xpc: the expConfig dataclass; same useage as complete_activation_test.
        verbose: boolean detailing whether to provide details over current execution cycle.
        categories: tuple of strings containing which categories (e.g grad, testloss, testpreds) are desired to evaluate.

    Returns:
        total_activation_dfs: a dictionary containing for each triple combination of evaluation type, category and
        measurement type, the corresponding dataframe of all associated results. 
        Each dataframe stores an agg-type test-type combination, e.g "('log_average', 'mean')" as a column name.
        
        expConfig: the modified experimentParams for the LS alpha sensitivity test, for future use.
    """
    
    if len(xpc.activations) != 1 :
        print(f"Got more than 1 activation for sensitivity test; using first choice. Please select only one.")
    
    base_activation = xpc.activations[0] 
    
    # Measuring alpha selection over entire domain - equal spacing for most representative results
    alphas = np.linspace(0, 1, xpc.n_alphas)
    activations = []
    activation_names = []
    
    for alpha in alphas.tolist() : 
        
        activation = to_LS(base_activation, alpha, learnable = False)     
        activation_name = f"$\\alpha = {alpha:.2f}$" # So we can read it from the plot
        
        activations.append(activation)
        activation_names.append(activation_name)
    
    new_xpc = replace(xpc, activations = activations, activation_names = activation_names)
    return complete_activation_test(new_xpc, verbose = verbose), new_xpc


def time2threshold_test(aresults : activationResults, eval_type : str, category : str, reducer : str, 
                   nthresholds : int = 50, ascending : bool = True) -> pd.DataFrame :
    
    """Compute the time-to-threshold test for all measured activations using a given category and reducer type, given 
    test results. This test measures the number of epochs required to reach each threshold in the category over a given set 
    of thresholds, calculated dynamically via the minimum and maximum values for that category. The thresholds represent the
    x-axis (independent variable), and the number of epochs represents the y-axis (dependent measured variable). 
    
    Thresholds are calculated as uniform separation points between the minimum and maximum threshold values, using 
    np.linspace to compute boundaries. This always uses epochs, not any other form of measurement. This means it works 
    for any category, since all categories are required to at least track the "epochs" measure type. 
    
    Params:
        aresults: the ActivationResults object acquired from complete_activation_test or a variant. 
        category: the category of data to compare thresholds for, e.g test loss or activation gradients.
        reducer: the aggregator function used to collapse all other dimensions into the epoch dimension. Usually "mean".
        nthresholds: how many threshold data points to use in calculation and thus form the x-axis. Defaults to 50.
        ascending: whether we want thresholds to be exceeded, or subceeded. E.g for test loss we want lower values.

    Returns:
        pd.DataFrame: the resulting threshold DataFrame containing epoch convergence data for each activation (column-format).
    """
    
    # Maps each position to its corresponding epoch, since there are far more epochs than recorded epochs (positions)
    positions2epochs = np.asarray(sampling_indices(config["epochs"], config["max_recorded_samples"]) + [np.nan])
    
    # Already starts in this order with query structure, but this guarantees it as an assumption
    table = aresults.query(eval_type, category, "epochs", reducer, "mean")[["activation", "position", "val"]]
    table.sort_values(by = ["activation", "position"], inplace = True, axis = 0) 
    
    avs = table["val"]
    bounds = (avs.min(), avs.max()) if ascending else (avs.max(), avs.min())
    thresholds = np.linspace(*bounds, nthresholds)
    
    # This makes every activation its own column which is vital for plotting later and for rest of computation
    # It will never produce NaN values because every activation has exactly the same number of epochs as every other
    activ_table = table.pivot(index = "position", columns = "activation", values = "val")
        
    # Shape (n_positions, n_activations, n_thresholds), tracks when first exceeded each threshold
    if ascending :
        compared2thresh = activ_table.to_numpy()[:, :, None] >= thresholds[None, None, :] 
    else : 
        compared2thresh = activ_table.to_numpy()[:, :, None] <= thresholds[None, None, :] 
    
    # Argmax will grab the integer index of the very first epoch to exceed/subceed the given threshold
    first_epochs2exceed = np.argmax(compared2thresh, axis = 0) # Now of shape (n_activations, n_thresholds)
    
    # If we don't update cases where it never passed a thresh, it will give a score of 0 which gives false impression
    never_exceeded = ~np.any(compared2thresh, axis = 0) 
    first_epochs2exceed[never_exceeded] = -1 #  Give it -1 index, which maps to nan in epoch_index_nmaes

    # Converts epoch indices to epoch names (although the same in 99.9% of cases). One bin reserved for never-reachers
    epoch_index_names = np.array(activ_table.index.tolist() + [-1])
    
    # In theory could just use first_epochs2exceed.T since it's already an integer array, but this is more robust
    # Double indexing; map position indices to true positions, then map those positions to the real epochs
    final_thresh_data = positions2epochs[epoch_index_names[first_epochs2exceed].T] 
    thresh_df = pd.DataFrame(data = final_thresh_data, index = thresholds, columns = activ_table.columns)
    thresh_df.index.name = "threshold"
        
    return thresh_df



def complete_time2threshold_test(aresults : activationResults, 
                                 nthresholds : int = 50, ascending : bool = True) -> dict[tuple[str, str, str], pd.DataFrame] :
    
    """Same as time2threshold_test, but iterates over all valid triples of (eval_type, category, reducer).
    
    Params:
        aresults: the activation results DataFrame to grab data from.
        nthresholds: how many thresholds in the resulting DataFrames.
        ascending: whether we want to exceed thresholds or subceed them. For granular control, please use time2threshold_test.

    Returns:
        dict[tuple[str, str, str], pd.DataFrame]: the dictionary mapping triples of (eval_type, category, reducer) to 
        threshold dataframes for each activation. See time2threshold_test for more documentation on that.
    """
    
    time2threshold_dict = {}
    with Progress() as progress : # Quite a few results so need to track for user satisfaction
        thresh_work = progress.add_task("Threshold experiment progress:" , total = len(aresults.results) )
        
        for eval_type, category, measure_type in aresults.results.keys() :
            triple_format = (eval_type, category, measure_type)   
            
            if measure_type != "epochs" : 
                progress.console.log(f"Result class {triple_format} not of measure type \"epochs\". Skipping.")
                progress.advance(thresh_work, 1)    
                continue # We never do time2threshold over any measure type other than epochs
            
            progress.console.log(f"Now calculating thresh data for result {triple_format}")
            reducers = [col[0] for col in aresults.results[triple_format].columns # Not all cols are tuples
                        if isinstance(col, tuple) and col[1] == "mean" ] # kf_reducer always mean, others do not make sense
            
            for reducer in reducers :
                thresh_df = time2threshold_test(aresults, eval_type, category, reducer, nthresholds, ascending)
                time2threshold_dict[(eval_type, category, reducer)] = thresh_df
            
            progress.advance(thresh_work, 1)    

    return time2threshold_dict

