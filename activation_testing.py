import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Any, Callable
from dataclasses import replace
from rich.progress import Progress

# CUSTOM
from support.processing_helpers import (sampling_indices, dfs_settings2tensors, 
                                        get_number_of_features_and_classes)
from support.config import config, set_seed
from support.torch_reducers import arithmetic_mean
from support.parsing_helpers import safe_asdict
from dataclass_objects.config_objects import expConfig
from dataclass_objects.input_objects import expInput, testInput
from dataclass_objects.result_objects import experimentResult, activationResults
from categories.category_registry import category_registry, categoryExperimentLogger
from networks import ActivationNetwork
from activations import to_LS


def experiment(xpi : expInput) -> experimentResult : 
    
    """
    Main experiment code for the project. Takes in tensors, model, loss, and metadata and returns result as a
    simple experimentResult dataclass for ease of use. Ideal for passing in expConfig dataclass as input.
    
    This is the main driver function for multiple experiment classes; please be careful when 
    modifying or removing functionality.
    
    Params:
        xpi: the main input params class for this function, containing all of the attributes below.
        device_thresh: number of samples before swapping to GPU
        
    Returns:
        experimentResult : stores all the captured data for future use.
    """
    
    xpi.save_state()
    xpi.to_device(xpi.preferred_device, strict = False)
    record_epochs = set(sampling_indices(xpi.epochs, xpi.n_captures))
    
    # Used for data recording
    logger = categoryExperimentLogger(xpi, categories = xpi.categories)
    
    record_index = 0
    for epoch in range(xpi.epochs) :
        xpi.anet_model.record(epoch in record_epochs)
        
        xpi.anet_model.train()
        for X_train_batch, Y_train_batch in xpi.training_dataloader :
            # Must be on same device to do processing with
            X_train_batch = X_train_batch.to(xpi.device)
            Y_train_batch = Y_train_batch.to(xpi.device)
            
            xpi.optim.zero_grad()
            predictions = xpi.anet_model(X_train_batch)
            loss : torch.Tensor = xpi.target_loss(predictions, Y_train_batch)
            loss.backward()
            xpi.optim.step()
        
        # Only capture the data at specified record times
        if not xpi.anet_model.recording : continue
            
        xpi.anet_model.eval()
        with torch.no_grad() :
            logger.record(record_index)
            record_index += 1 
            
    xpi.reload_state(switch_device = False)        
    
    # gm = 2D (epochs, parameters), AL = 3D (epochs, layers, neurons), TL = 1D (epochs), TP = 2D (epochs, n_test)
    return logger.result


def experiment_from_df(df_train : pd.DataFrame, df_test : pd.DataFrame, model : ActivationNetwork,
                       labels : str | list[str], loss : nn.Module = nn.CrossEntropyLoss(),
                       feature_transforms : tuple[tuple[list[str], Any], ...] = (),
                       label_transforms : tuple[tuple[list[str], Any], ...] = (),
                       dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long), 
                       epochs : int = 500, batch_size : int = -1, max_recorded_samples : int = -1,
                       categories : tuple[str, ...] = ("grad",)) -> experimentResult :
    
    """
    Same as experiment() but taken directly from the dataframe to minimise boilerplate code. Also excellent for 
    adapting with expConfig() dataclass. 

    Params: 
        df_train: Dataframe containing train data.
        df_test: Dataframe containing test data.
        model: model for use in experiment. 
        labels: subset of columns to be predicted on.
        loss: loss function.
        feature_transforms: n-tuple of 2-tuples, where the former is the list of columns to transform and the latter is the 
        transformer to apply for those columns.
        label_transforms: same as feature_transforms but for label columns. Note that for either case, columns not specified 
        will be left untransformed.
        dtypes: 2-tuple containing data types of features and of labels after transformation. Supports one type per group.
        epochs: number of complete sweeps of X_train_tensor and Y_train_tensor to perform to train the model.
        batch_size: number of training examples to use per gradient descent step. Defaults to -1 (all).
        max_recorded_samples: maximum number of training steps captured in experimentResult. Defaults to -1 (all).
        
    Returns:
        experimentResult: stores all the captured data for future use
    """
    
    X_train, X_test, Y_train, Y_test = dfs_settings2tensors(df_train, df_test, 
                                                            feature_transforms, label_transforms, 
                                                            labels, dtypes)
    
    return experiment(expInput(X_train, X_test, Y_train, Y_test, model, target_loss = loss, 
                      epochs = epochs, batch_size = batch_size, max_recorded_samples = max_recorded_samples, 
                      categories = categories))

def skf_crossval(df : pd.DataFrame, model : ActivationNetwork, labels : str | list[str], 
                loss : nn.Module,
                feature_transforms : tuple[tuple[list[str], Any], ...] = (), 
                label_transforms : tuple[tuple[list[str], Any], ...] = (),
                dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long),
                kfold_k : int = 10, epochs : int = 500, 
                batch_size : int = -1, max_recorded_samples : int = -1,
                categories : tuple[str, ...] = ("grad",)) -> experimentResult :
    
    """
    Perform Stratified K-Fold (SKF) cross-validation on a dataframe. Highly compatible with 
    expConfig() dataclass using safe_asdict helper function. 
    
    Params:
        df: the dataframe to perform SKF with. Please ensure no test samples are stored here.
        model: the model to perform SKF with.
        labels: subset of columns to be predicted on.
        kfold_k: number of folds to perform stratified KFold on.
        epochs: number of complete sweeps of X_train_tensor and Y_train_tensor to perform to train the model.
        feature_transforms: n-tuple of 2-tuples, where the former is the list of columns to transform and the latter
        is the transformer to apply for those columns.
        label_transforms: same as feature_transforms but for label columns. Note that for either case, columns not 
        specified will be left untransformed.
        dtypes: 2-tuple containing data types of features and of labels after transformation. Supports one type per group.  
        batch_size: number of training examples used per gradient descent step. Defaults to -1 (all).
        max_recorded_samples: maximum number of training steps captured in experimentResult. Defaults to -1 (all).

    NOTE: For future use, note that the k-fold dimension must *always* be the last one, or the program breaks.

    Returns:
        experimentResult: stores all folds and captured data for use. 
    """
    
    skf = StratifiedKFold(n_splits = kfold_k, random_state = config["seed"], shuffle = True)

    exp_results = []
    X_train = df.drop(columns = labels)
    Y_train = df[labels] 
    
    # Store the train and test data indices for use to calculate metrics later
    kfold_data = []

    for train_index, test_index in skf.split(X_train, Y_train) :
        
        train_data : pd.DataFrame = df.iloc[train_index]
        test_data : pd.DataFrame = df.iloc[test_index]
        fold_result = experiment_from_df(train_data, test_data, model, labels, 
                               feature_transforms = feature_transforms, 
                               label_transforms = label_transforms, 
                               dtypes = dtypes, epochs = epochs, 
                               batch_size = batch_size, max_recorded_samples = max_recorded_samples, loss = loss,
                               categories = categories) 
        
        exp_results.append(fold_result)
        kfold_data.append((train_data, test_data))

    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return experimentResult(exp_results, metadata = {"kfold_data" : kfold_data})

def multiseed_test_from_df(df_train : pd.DataFrame, df_test : pd.DataFrame, model : ActivationNetwork,
                       labels : str | list[str], loss : nn.Module = nn.CrossEntropyLoss(),
                       feature_transforms : tuple[tuple[list[str], Any], ...] = (),
                       label_transforms : tuple[tuple[list[str], Any], ...] = (),
                       dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long), 
                       epochs : int = 500, batch_size : int = -1, max_recorded_samples : int = -1,
                       categories : tuple[str, ...] = ("grad",), n_testseeds : int = 10) -> experimentResult :
    
    """Same as experiment_from_df, but uses multiple different testseeds to repeat the experiment, then averages out
    the results. For long data (e.g metrics), will use the mode, and for all other data will use the standard NaN-aware mean. 
    
    Params:
        n_testseeds: how many testseeds to run.
        For all other parameters, please refer to experiment_from_df as they are identical.

    Returns:
        experimentResult: the test data results.
    """
    
    seeds = np.random.randint(42, 2026, n_testseeds).tolist()
    exp_results = []
    
    for seed in seeds :
        set_seed(seed)
        test_result = experiment_from_df(df_train, df_test, model, labels, loss, 
                                         feature_transforms, label_transforms, dtypes,
                                         epochs, batch_size, max_recorded_samples, categories)
        exp_results.append(test_result)
    
    # Combining results together via keyname already covered with experimentResult constructor, re-use here
    exp_result = experimentResult(exp_results, metadata = {"seeds" : seeds})
    averaged_exp_result_dict = {}
    
    for result_str, result in exp_result.results.items() :
        match result.dtype :
            case torch.long : # Using mean on continuous data could output errors, e.g 1 and 3 --> prediction 2
                avg_result = torch.mode(result, dim = -1).values.to(torch.long)
            case _ :
                avg_result = torch.nanmean(result, dim = -1)
        averaged_exp_result_dict[result_str] = avg_result.to(result.dtype)
        
    # Assign to both hidden and default results object for consistency
    exp_result.results = exp_result._results = averaged_exp_result_dict
    
    # Side effects not permitted 
    set_seed(config["seed"])
    
    return exp_result
    
    

def evaluate_activation_results(xpc : expConfig, exp_result : experimentResult, tester : Callable,
                            typeof_result : tuple[str, ...]) -> pd.DataFrame : 
    
    # Unpack the metadata components from the type of result we have 
    eval_type, category, measure_type = typeof_result
    
    # Get the correct category params dataclass object, then get the data to evaluate - either train or test
    data = exp_result.results[category]
    # r[eval_type] gets the right object from r ("train" vs "test"), then select the right data 
    # from the experimentResult r.results, which is a dictionary of categories to data tensors 
    
    expected_ndims = len(data.shape)
    if eval_type == "train" :
        aggfuncs, aggfunc_names = xpc.kf_reducers, xpc.kf_reducer_names
        expected_ndims -= 1 # We already have kfold dimension so subtract 1 dim to compensate
    else : # No point aggregating over a single fold if it's test data; 0 variance    
        nameof_arithmetic_mean = xpc.kf_reducer_names[xpc.kf_reducers.index(arithmetic_mean)]
        aggfuncs, aggfunc_names = ((arithmetic_mean,), [nameof_arithmetic_mean])

    # Encapsulate into test input dataclass to avoid path dependence; store misc. vitals in metadata
    test_input = testInput(data, 
                            xpc.reducers, xpc.reducer_names, 
                            aggfuncs, aggfunc_names, 
                            measure_type = measure_type, 
                            expected_ndims = expected_ndims, 
                            metadata = exp_result.metadata,
                            xpc = xpc)

    results_df = tester(test_input) 
    return results_df


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