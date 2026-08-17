import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Any, Callable
from dataclasses import replace
from rich.progress import Progress
from random import sample

# CUSTOM
from support.processing_helpers import (sampling_indices, dfs_settings2tensors, 
                                        get_number_of_features_and_classes)
from support.config import config, set_seed
from support.torch_reducers import arithmetic_mean
from support.parsing_helpers import safe_asdict
from dataclass_objects.config_objects import expConfig
from dataclass_objects.input_objects import expInput, testInput
from dataclass_objects.result_objects import ExperimentResult
from categories.category_registry import categoryExperimentLogger
from networks import ActivationNetwork

def experiment(xpi : expInput) -> ExperimentResult : 
    
    """
    Main experiment code for the project. Takes in tensors, model, loss, and metadata and returns result as a
    simple ExperimentResult dataclass for ease of use. Ideal for passing in expConfig dataclass as input.
    
    This is the main driver function for multiple experiment classes; please be careful when 
    modifying or removing functionality.
    
    Params:
        xpi: the main input params class for this function, containing all of the attributes below.
        device_thresh: number of samples before swapping to GPU
        
    Returns:
        ExperimentResult : stores all the captured data for future use.
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
                       categories : tuple[str, ...] = ("grad",)) -> ExperimentResult :
    
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
        max_recorded_samples: maximum number of training steps captured in ExperimentResult. Defaults to -1 (all).
        
    Returns:
        ExperimentResult: stores all the captured data for future use
    """
    
    X_train, X_test, Y_train, Y_test = dfs_settings2tensors(df_train, df_test, 
                                                            feature_transforms, label_transforms, 
                                                            labels, dtypes)
    
    exp = experiment(expInput(X_train, X_test, Y_train, Y_test, model, target_loss = loss, 
                      epochs = epochs, batch_size = batch_size, max_recorded_samples = max_recorded_samples, 
                      categories = categories))

    # Attach the original train data so we can grab it later without having to repeat input processing
    exp.metadata["data"] = {"X_train" : X_train, "X_test" : X_test, "Y_train" : Y_train, "Y_test" : Y_test}

    return exp

def skf_crossval(df : pd.DataFrame, model : ActivationNetwork, labels : str | list[str], 
                loss : nn.Module,
                feature_transforms : tuple[tuple[list[str], Any], ...] = (), 
                label_transforms : tuple[tuple[list[str], Any], ...] = (),
                dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long),
                kfold_k : int = 10, epochs : int = 500, 
                batch_size : int = -1, max_recorded_samples : int = -1,
                categories : tuple[str, ...] = ("grad",)) -> ExperimentResult :
    
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
        max_recorded_samples: maximum number of training steps captured in ExperimentResult. Defaults to -1 (all).

    NOTE: For future use, note that the k-fold dimension must *always* be the last one, or the program breaks.

    Returns:
        ExperimentResult: stores all folds and captured data for use. 
    """
    
    skf = StratifiedKFold(n_splits = kfold_k, random_state = config["seed"], shuffle = True)

    exp_results = []
    X_train = df.drop(columns = labels)
    Y_train = df[labels] 
    
    # Store the train and test data indices for use to calculate metrics later
    kfold_data = []

    for train_index, test_index in skf.split(X_train, Y_train) :
        
        fold_result = experiment_from_df(df.iloc[train_index], df.iloc[test_index], model, labels, 
                               feature_transforms = feature_transforms, 
                               label_transforms = label_transforms, 
                               dtypes = dtypes, epochs = epochs, 
                               batch_size = batch_size, max_recorded_samples = max_recorded_samples, loss = loss,
                               categories = categories) 
        
        exp_results.append(fold_result)
        
        data = fold_result.metadata["data"] # Need this for reconstruction later in metrics
        kfold_data.append((data["X_train"], data["X_test"], data["Y_train"], data["Y_test"]))

    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return ExperimentResult(exp_results, metadata = {"kfold_data" : kfold_data})

def multiseed_test_from_df(df_train : pd.DataFrame, df_test : pd.DataFrame, model : ActivationNetwork,
                       labels : str | list[str], loss : nn.Module = nn.CrossEntropyLoss(),
                       feature_transforms : tuple[tuple[list[str], Any], ...] = (),
                       label_transforms : tuple[tuple[list[str], Any], ...] = (),
                       dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long), 
                       epochs : int = 500, batch_size : int = -1, max_recorded_samples : int = -1,
                       categories : tuple[str, ...] = ("grad",), n_testseeds : int = 10) -> ExperimentResult :
    
    """Same as experiment_from_df, but uses multiple different testseeds to repeat the experiment, then averages out
    the results. For long data (e.g metrics), will use the mode, and for all other data will use the standard NaN-aware mean. 
    
    Params:
        n_testseeds: how many testseeds to run.
        For all other parameters, please refer to experiment_from_df as they are identical.

    Returns:
        ExperimentResult: the test data results.
    """
    
    if n_testseeds <= 0 : 
        raise ValueError(f"Must use at least 1 seed in multiseed test data experiment")
    
    seeds = sample(range(42, 2026), n_testseeds)
    exp_results = [] 
    
    # We will be using the same train-test data, only the seed itself changes. Define once to avoid redundancy in processing
    traintest_data = dfs_settings2tensors(df_train, df_test, feature_transforms, label_transforms, labels, dtypes)
    exp_input_object = expInput(*traintest_data, model, target_loss = loss, epochs = epochs, batch_size = batch_size, 
                                max_recorded_samples = max_recorded_samples, categories = categories)

    for seed in seeds : # This is the only "running" part of the entire function.
        set_seed(seed)
        test_result = experiment(exp_input_object)
        exp_results.append(test_result)
    
    # Combining results together via keyname already covered with ExperimentResult constructor, re-use here
    exp_result = ExperimentResult(exp_results, metadata = {"seeds" : seeds, "kfold_data" : [traintest_data]})
    averaged_exp_result_dict = {} # Wrap the train, test data in list and call it kfold to avoid branching in metrics.py
    
    # This part aggregates the extra final dimension of seed results, which otherwise would be treated as kfold dim
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
    
    
def evaluate_activation_results(xpc : expConfig, exp_result : ExperimentResult,
                                tester : Callable[[testInput], pd.DataFrame], typeof_result : tuple[str, ...]
                                ) -> pd.DataFrame : 
    
    """Main testing function on experimental results. Given the experiment configuration, result, tester and the
    result metadata, calculate the test results and return as a Pandas DataFrame object.

    Params:
        xpc: expConfig dataclass containing all experiment metadata including the relevant reducers for extracting the DF.
        exp_result: experimentResult dataclass containing the results of the experiment.
        tester: the function to test the result with. Must return a DataFrame.
        typeof_result: tuple containing eval_type, category and measure_type information to perform the corresponding test.

    Returns:
        DataFrame: DataFrame of test results. 
    """
    
    # Unpack the metadata components from the type of result we have 
    eval_type, category, measure_type = typeof_result
    
    # Get the correct category params dataclass object, then get the data to evaluate - either train or test
    data = exp_result.results[category]
    # r[eval_type] gets the right object from r ("train" vs "test"), then select the right data 
    # from the ExperimentResult r.results, which is a dictionary of categories to data tensors 
    
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