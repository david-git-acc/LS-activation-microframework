import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Any, Callable
from dataclasses import replace
import copy
from rich.progress import Progress

# CUSTOM
from helpers import *
from dataclass_objects import expConfigParams, monitorParams, expInputParams, experimentResult
from category_functions import category_registry, categoryExperimentLogger
from networks import ActivationNetwork
from activations import LS

def experiment(xpi : expInputParams) -> experimentResult : 
    
    """
    Main experiment code for the project. Takes in tensors, model, loss, and metadata and returns result as a
    simple experimentResult dataclass for ease of use. Ideal for passing in expConfigParams dataclass as input.
    
    This is the main driver function for multiple experiment classes; please be careful when 
    modifying or removing functionality.
    
    Params:
        xpi: the main input params class for this function, containing all of the attributes below.
        X_train_tensor : n x d training feature matrix.
        X_test_tensor : n_test x d testing feature matrix.
        Y_train_tensor : 1 x n or n x 1 training label matrix.
        Y_test_tensor : 1 x n_test or n_test x 1 testing label matrix.
        anet_model : the model to perform the experiment with.
        my_loss : the loss function to evaluate the model.
        epochs : number of complete sweeps of X_train_tensor and Y_train_tensor to perform to train the model.
        lr : constant learning rate value. In theory, you could pass in a variable learning rate here (not recommended).
        batch_size : number of training examples to use per gradient descent step. Defaults to -1 (all examples per step).
        max_samples : maximum number of training steps to be recorded and captured in experimentResult. Defaults to -1 (all). 
        
    Returns:
        experimentResult : stores all the captured data for future use.
    """
    
    xpi.save_state()
    optim = torch.optim.Adam(xpi.anet_model.parameters(), lr = xpi.lr)
    record_epochs = set(sampling_indices(xpi.epochs, xpi.n_captures))
    
    # Used for data recording
    logger = categoryExperimentLogger(xpi, categories = "all")
    
    record_index = 0
    for epoch in range(xpi.epochs) :
        
        xpi.anet_model.train()
        for X_train_batch, Y_train_batch in xpi.training_dataloader :
            optim.zero_grad()
            predictions = xpi.anet_model(X_train_batch)
            loss : torch.Tensor = xpi.my_loss(predictions, Y_train_batch)
            loss.backward()
            optim.step()
        
        # Only capture the data at specified record times
        if epoch not in record_epochs : continue
            
        xpi.anet_model.eval()
        with torch.no_grad() :
            logger.record(record_index)
            record_index += 1 
            
    xpi.reload_state()        
    
    # gm = 2D (epochs, parameters), AL = 3D (epochs, layers, neurons), TL = 1D (epochs), TP = 2D (epochs, n_test)
    return logger.result


def experiment_from_df(df_train : pd.DataFrame, df_test : pd.DataFrame, model : ActivationNetwork,
                       labels : str | list[str], loss : nn.Module = nn.CrossEntropyLoss(),
                       feature_transforms : tuple[tuple[list[str], Any], ...] = (),
                       label_transforms : tuple[tuple[list[str], Any], ...] = (),
                       dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long), 
                       epochs : int = 500, batch_size : int = -1, max_samples : int = -1 ) -> experimentResult :
    
    """
    Same as experiment() but taken directly from the dataframe to minimise boilerplate code. Also excellent for 
    adapting with expConfigParams() dataclass. 

    Params: 
        df_train: Dataframe containing train data.
        df_tes: Dataframe containing test data.
        model: model for use in experiment. 
        labels: subset of columns to be predicted on.
        loss: loss function.
        feature_transforms: n-tuple of 2-tuples, where the former is the list of columns to transform and the latter is the 
        transformer to apply for those columns.
        label_transforms: same as feature_transforms but for label columns. Note that for either case, columns not specified 
        will be left untransformed.
        dtypes: 2-tuple containing data types of features and of labels after transformation. Only supports one type per group.
        epochs: number of complete sweeps of X_train_tensor and Y_train_tensor to perform to train the model.
        batch_size: number of training examples to use per gradient descent step. Defaults to -1 (all training examples per step)
        max_samples: maximum number of training steps to be recorded and captured in experimentResult. Defaults to -1 (all). 
        
    Returns:
        experimentResult: stores all the captured data for future use
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
    
    return experiment(expInputParams(X_train, X_test, Y_train, Y_test, model, my_loss = loss, 
                      epochs = epochs, batch_size = batch_size, max_samples = max_samples))

        
        
def skf_crossval(df : pd.DataFrame, model : ActivationNetwork, labels : str | list[str], 
                loss : nn.Module,
                feature_transforms : tuple[tuple[list[str], Any], ...] = (), 
                label_transforms : tuple[tuple[list[str], Any], ...] = (),
                dtypes : tuple[torch.dtype, torch.dtype] = (torch.float32, torch.long),
                kfold_k : int = 10, epochs : int = 500, 
                batch_size : int = -1, max_samples : int = -1) -> experimentResult :
    
    """
    Perform Stratified K-Fold (SKF) cross-validation on a dataframe. Highly compatible with 
    expConfigParams() dataclass using safe_asdict helper function. 
    
    Params:
        df: the dataframe to perform SKF with. Please ensure no test samples are stored here.
        model: the model to perform SKF with.
        labels: subset of columns to be predicted on.
        kfold_k: number of folds to perform stratified KFold on.
        epochs: number of complete sweeps of X_train_tensor and Y_train_tensor to perform to train the model.
        feature_transforms: n-tuple of 2-tuples, where the former is the list of columns to transform and the latter is the 
        transformer to apply for those columns.
        label_transforms: same as feature_transforms but for label columns. Note that for either case, columns not specified 
        will be left untransformed.
        dtypes: 2-tuple containing data types of features and of labels after transformation. Only supports one type per group.  
        batch_size: number of training examples to use per gradient descent step. Defaults to -1 (all training examples per step)
        max_samples: maximum number of training steps to be recorded and captured in experimentResult. Defaults to -1 (all).

    NOTE: For any future use or modification, note that the k-fold dimension must *always* be the last one, or the program breaks.

    Returns:
        experimentResult: stores all folds and captured data for use. 
    """
    
    skf = StratifiedKFold(n_splits = kfold_k, random_state = config["seed"], shuffle = True)

    exp_results = []
    X_train = df.drop(columns = labels)
    Y_train = df[labels] 
    
    for train_index, test_index in skf.split(X_train, Y_train) :
        
        fold_result = experiment_from_df(df.iloc[train_index], df.iloc[test_index], model, labels, 
                               feature_transforms = feature_transforms, 
                               label_transforms = label_transforms, 
                               dtypes = dtypes, epochs = epochs, 
                               batch_size = batch_size, max_samples = max_samples, loss = loss) 
        
        exp_results.append(fold_result)

    # gms = 3D (epochs, parameters, folds), kfl = 2D (epochs, folds), kfold_tps = 3D (epochs, n_test, folds)
    return experimentResult(exp_results)

  
  
def post_experiment_test_grad(gms : torch.Tensor, over : str = "epochs",
                          test_suite : tuple[Callable, ...] = (), test_columns : list[str] = [], 
                          kfold_aggfuncs : tuple[Callable, ...] = (arithmetic_mean,), 
                          kfold_columns : list[str] = ["mean"],
                          expected_ndims : int = 2) -> pd.DataFrame :
    
    """
    Perform the function test suite on a designated set of test functions with k-folds, then collapses over the k-folds using
    an aggregation function (typically mean) and returns results as a Pandas dataframe.
    
    Params:
        gms : list of gradient matrices over folds (3D: (epochs, parameters, folds) - although it need not be this shape)
        over: dimension to check over. Can be set to a different axis manually.
        test_suite: list of functions to test on.
        test_columns: names of each test. If no value given, uses the function names.
        kfold_aggfuncs: the aggregation functions to collapse a dimension over. Always becomes mean() if number of folds = 1.
        expected_ndims : number of dimensions that the data is originally meant to be in (before folds). Used for validation.
        
    NOTE: This function is the main function for post experiment testing; testloss and testpreds rely on this one. Also,
    remember that the last dimension must always be the kfold dimension, or bugs will occur - silently or not.


    Returns:
        result_df: Pandas dataframe containing results. Each column is a different agg-type test-type combination.
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
    
    # For each test function we compute the result and collapse all other dimensions using it, to get a 2D array (dim, folds)
    for i, test_func in enumerate( test_suite ) : 
        result = test_func(mp.X, dim = dim) # Make sure all test suite functions is NaN-aware + can handle any subset of dims
        
        # Then we collapse the fold dimension in different ways; these are important, will be covered later.
        for j, kfold_aggfunc in enumerate( mp.kfold_aggfuncs ) :
        
            kfold_dim = len(result.size()) - 1
        
            collapsed_result = kfold_aggfunc(result, dim = kfold_dim )
            data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
            test_results.append(data)
            
            df_column_name = (mp.test_function_names[i], mp.kfold_columns[j])
            df_columns.append(df_column_name)
    
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    result_df = pd.DataFrame(test_results, columns = df_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = over[:-1] # Kill the "s", we view singularly
        
    return result_df    

def post_experiment_test_testloss(tl : torch.Tensor, over : None = None,
                          test_suite : None = None, test_columns : list[str] = [], 
                          kfold_aggfuncs :  tuple[Callable, ...] = (arithmetic_mean,), 
                          kfold_columns : list[str] = ["mean"]) -> pd.DataFrame :
    
    """
    Same as post_experiment_test_grad, but for testloss. Identical logic.
    Note that test_suite, over and test_columns are deprecated because only 1 dimension is supported.

    Returns:
        results_df: dataframe of results. Each column is a different agg-type.
    """
    
    return post_experiment_test_grad(tl, "epochs", ( testloss_dummy, ), ["test loss"], 
                                     kfold_aggfuncs, kfold_columns, expected_ndims = 1)

def post_experiment_test_testpreds(tps : torch.Tensor, over : str = "test_samples",
                          test_suite : tuple[Callable, ...] = (), test_columns : list[str] = [], 
                          kfold_aggfuncs : tuple[Callable, ...] = (arithmetic_mean,), 
                          kfold_columns : list[str] = ["mean"]) -> pd.DataFrame :
    
    """
    Same as post_experiment_test_grad, but for test predictions (testpreds). Identical logic.
    
    Returns:
        result_df: Pandas dataframe containing results. Each column is a different agg-type test-type combination.
    """
        
    return post_experiment_test_grad(tps, over, test_suite, test_columns, kfold_aggfuncs, kfold_columns)


def complete_activation_test(exp_params : expConfigParams, verbose : bool = False) -> dict[tuple[str,str,str], pd.DataFrame] :
    
    """
    Orchestrator / god function to perform entire experiment, from training to kfold and testing given a set of 
    expConfigParams. Designed to minimise boilerplate and facilitate ease of use + modularity. 
    
    Params:
        exp_params: the expConfigParams dataclass containing all important data about the experiment. 
        verbose: boolean detailing whether to provide details over current execution cycle.

    NOTE: if no category is specified in expConfigParams, it will use all categories. Remember to add category parameters
    to the expConfigParams dataclass inside dataclass_objects.py.
    
    Category selection is a tuple containing category parameters, each of which stores the triple combination of 
    category name, the associated tester function over that category (not to be confused with test functions,
    which operate directly over data (e.g mean)), and the tuple of all valid measurement types. S

    Returns:
        total_activation_dfs: a dictionary containing for each triple combination of evaluation type, category and
        measurement type, the corresponding dataframe of all associated results. 
        Each dataframe stores an agg-type test-type combination, e.g "('log_average', 'mean')" as a column name.
    """
    
    category_params = {cat : category_registry[cat] for cat in exp_params.categories}
    
    n_features, n_classes = get_number_of_features_and_classes(exp_params.df_train, exp_params.labels)

    # Add to these incrementally and then concatenate at the end to turn into dataframes.
    total_activation_dfs = {(eval_type, category.name, measure_type) : [] 
                            for eval_type in ("train", "test")
                            for category in category_params.values()
                            for measure_type in category.measure_types }
    
    with Progress() as progress :
        work = progress.add_task("Experiment progress:", total = len(exp_params.activations) * len(total_activation_dfs))
        
        for activation_index, activation in enumerate( exp_params.activations ) : 

            network = exp_params.network_type(activation, n_inputs = n_features, n_outputs = n_classes)
            activation_name =  exp_params.activation_names[activation_index] 
            
            net_act_str = f"[N: {network.__class__.__name__}, A: {activation.__class__.__name__}]"  
            if verbose : progress.console.log(f"Executing configuration: {net_act_str}")

            # Pass in params dataclass directly because function signature may become arbitrarily long with more additions
            r = {"train" : skf_crossval(**safe_asdict(exp_params, skf_crossval), model = network, df = exp_params.df_train),
                 "test" : experiment_from_df(**safe_asdict(exp_params, experiment_from_df), model = network)}

            for eval_type, category, measure_type in total_activation_dfs:
                if verbose : progress.console.log(f" -> {eval_type.title()}ing on {category} data over {measure_type}")
                
                if eval_type == "train" :
                    aggfuncs, aggfunc_names = exp_params.kfold_aggfuncs, exp_params.kfold_aggfunc_names
                else : # No point aggregating over a single fold if it's test data; 0 variance    
                    nameof_arithmetic_mean = exp_params.kfold_aggfunc_names[exp_params.kfold_aggfuncs.index(arithmetic_mean)]
                    aggfuncs, aggfunc_names = ((arithmetic_mean,), [nameof_arithmetic_mean])
                  
                # Get the correct category params dataclass object, then get the data to evaluate - either train or test
                c = category_params[category]
                data = r[eval_type].results[category]
                # r[eval_type] gets the right object from r ("train" vs "test"), then select the right data 
                # from the experimentResult r.results, which is a dictionary of categories to data tensors

                results_df = c.tester(data, over = measure_type, test_suite = exp_params.test_functions, 
                                      kfold_aggfuncs = aggfuncs, kfold_columns = aggfunc_names) 
                results_df["activation"] = activation_name # So we can keep track
            
                total_activation_dfs[(eval_type, category, measure_type)].append(results_df)
                progress.advance(work, 1)

    # Only concat at the end for speed
    total_activation_dfs = {df_type : pd.concat(df) for df_type, df in total_activation_dfs.items()}
    
    return total_activation_dfs


def LS_alpha_sensitivity_test(exp_params : expConfigParams, verbose : bool = True) -> dict[tuple[str, str, str], pd.DataFrame] :
    
    """
    Perform an alpha sensitivity test on an LS-converted activation function. Note that this implicitly assumes that
    the function specified is already in the S function family. An LS function exists in the form
    
    LS_{alpha}(f(x) E S) = alpha x + (1 - alpha) ( f(x) / f'(0) ) : alpha in (0,1)
    
    Creates as many equally spaced alphas as specified in the experiment parameters between 0 and 1 exclusive. Then,
    applies the standard complete activation test orchestrator function. Accepts the function from the 
    exp_params tuple of activation functions; if there is more than one, takes the first only. 

    Params:
        exp_params: the expConfigParams dataclass; same useage as complete_activation_test.
        verbose: boolean detailing whether to provide details over current execution cycle.
        categories: tuple of strings containing which categories (e.g grad, testloss, testpreds) are desired to evaluate.

    Returns:
        total_activation_dfs: a dictionary containing for each triple combination of evaluation type, category and
        measurement type, the corresponding dataframe of all associated results. 
        Each dataframe stores an agg-type test-type combination, e.g "('log_average', 'mean')" as a column name.
    """
    
    if len(exp_params.activations) != 1 :
        print(f"Got more than 1 activation for sensitivity test; using first choice. Please select only one.")
    
    base_activation = exp_params.activations[0] 
    
    # Measuring alpha selection over entire domain - equal spacing for most representative results
    alphas = np.linspace(0, 1, exp_params.n_alphas)
    activations = []
    activation_names = []
    
    for alpha in alphas.tolist() : 
        
        activation = LS(base_activation, alpha, learnable = False)     
        activation_name = f"$\\alpha ={alpha:.2f}$" # So we can read it from the plot
        
        activations.append(activation)
        activation_names.append(activation_name)
    
    new_experiment_params = replace(exp_params, 
                                    activations = activations,
                                    activation_names = activation_names)
    
    return complete_activation_test(new_experiment_params, verbose = verbose)


