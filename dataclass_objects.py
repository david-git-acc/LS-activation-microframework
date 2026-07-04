from dataclasses import dataclass, asdict, fields, field
import pandas as pd
from networks import ActivationNetwork
from typing import Any, Callable
import torch
from torch import nn
from helpers import (validate_activation_df_column_names, sampling_indices, get_n_colours, dummy_idfunc, 
                     category2measure_types, determine_plot_type, generate_plot_title, safe_set_params, arithmetic_mean,
                     is_hashable, smart_str)
import matplotlib.lines as mlines
import json
import hashlib
from math import ceil
from itertools import product


@dataclass
class experimentParams() :
    
    """
    Main dataclass for conducting activation experiments efficiently. This dataclass is designed
    to compare the effectiveness of different activation functions on a given network using a set of 
    test functions (test_functions) to marginalise unwanted dimensions and compare 1D arrays of results.
    
    Params:
        df_train: dataframe for train data.
        df_test: dataframe for test data.
        
        labels: list of columns that are to be predicted from features (presumed rest)
        
        network_type: type of neural network to work with for the experiment and perform predictions.
        
        loss: loss metric to judge predictions by. 
        
        feature_transforms: tuple of tuples, where the first entry of each tuple is the list of columns to be transformed,
        and the second entry is the transformer class to apply the transformation to each of the columns.
        
        label_transforms: same as feature_transforms, but for labels.
        
        test_functions: tuple of test functions to apply on finished results to marginalise unwanted dimensions. 
        activations: tuple of all activation functions to run in the experiment.
        
        kfold_aggfuncs: tuple of all aggregation functions to collapse folds over in K-fold crossvalidation, 
        e.g mean, variance, etc over folds. Note for test data it's interpreted as 1-fold crossvalidation, and only 
        mean is permitted for this.
        
        lr: constant learning rate. Can be changed to a variable one.
        
        kfold_k: number of folds to use in Kfold cross validation. 10 is the industry standard.
        
        n_alphas: number of different alpha values in the range (0,1) to use for LS sensitivity testing, with uniform spacing.
        E.g n_alphas = 5 would imply the alpha values (0.0, 0.25, 0.50, 0.75, 1.0).
        
        batch_size: size of train set used per iteration per epoch. Higher values increase gradient accuracy at the cost of time.
        
        max_samples: maximum number of samples to record (does not change number of epochs). Higher values create denser graphs.
        
        epochs: number of total training runs to apply to the neural network, where each epoch is a full pass of df_train.
        
        activation_names: list of display names for the activation functions, index-linked with activations.

        test_function_names: list of display names for the test functions. Again, index-linked with test_functions.
        
        kfold_aggfunc_names: list of display names for the aggregation functions, index-linked with kfold_aggfuncs.
        
        features_dtype: datatype of all features. Multiple datatypes for different features are not currently supported.
        
        labels_dtype: same as features_dtype but for labels.
        
        categories: tuple of all categories to test over. Defaults to all of them if none selected.

    """
    
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    labels : str | list[str]
    network_type : type[ActivationNetwork]
    loss : nn.Module
    feature_transforms : tuple[tuple[list[str], Any]]
    label_transforms : tuple[tuple[list[str], Any]]
    test_functions : tuple[Callable, ...]
    activations : list[nn.Module] | tuple[nn.Module, ...]
    kfold_aggfuncs : tuple[Callable, ...]
    lr : float = 0.001
    kfold_k : int = 10
    n_alphas : int = 5
    batch_size : int = -1 # full batch
    max_samples : int = -1 # no limit
    epochs : int = 500
    activation_names : list[str] = field(default_factory = list)
    test_function_names : list[str] = field(default_factory = list)
    kfold_aggfunc_names : list[str] = field(default_factory = list)
    features_dtype : torch.dtype = torch.float32
    labels_dtype : torch.dtype = torch.long
    categories : tuple[str, ...] = ("grad", "testloss", "testpreds")
    
    def __post_init__(self) :
        
        """
        Validation of all inputs to make sure the experiment runs smoothly and every activation has a valid name.
        """
        
        # YAML doesn't support tuples so convert in the program
        for potentially_list_attribute in ( "activations", "test_functions", "kfold_aggfuncs" ) :
            attr = getattr(self, potentially_list_attribute)
            if not isinstance(attr, tuple) :
                setattr(self, potentially_list_attribute, tuple(attr))
        
        # Placate the linter + consistency
        if isinstance(self.labels, str) : 
            self.labels = [self.labels]

        self.test_function_names = validate_activation_df_column_names(self.test_functions, self.test_function_names)
        self.activation_names = validate_activation_df_column_names(self.activations, self.activation_names)
        self.kfold_aggfunc_names = validate_activation_df_column_names(self.kfold_aggfuncs, self.kfold_aggfunc_names)
        
        # Mean is non-optional for experimentParams, we need it for when we collect results over test data
        if arithmetic_mean not in self.kfold_aggfuncs :
            self.kfold_aggfunc_names.append("mean")
            self.kfold_aggfuncs += (arithmetic_mean, )
    
    def savename(self , maxlen : int = 10) -> str :
        
        """Generates a deterministic savename for each experiment based on its deterministically string-able properties.
            For regular experiments, shows the network name, the initials of all activations used, then the signature hash.
            For LS alpha sensitivity experiments, shows the network name, the activation, number of alphas, and the hash.
            *Any* change in parameters or potentially the code structure itself will likely result in a different hash.

        Params:
            maxlen: defaults to 10, maximum length of the hash component.

        Returns:
            The savename for the experiment. Useful to track experiments and avoid experiment crowding in the main directory.
        """
        
        # Salt added on end
        strformat = str([ smart_str(v) for v in self.__dict__.values() if is_hashable(v)])
        as_str = json.dumps(strformat + "you and I, (nothing comes easy) * 2", sort_keys = True).encode() 
        
        hash_monstrosity = hashlib.sha256(as_str).hexdigest()
        
        if len(self.activation_names) == 1 :
            plus_extra = f"{self.activation_names[0]}_alpha[{self.n_alphas}]"
        else :
            plus_extra = "_".join([name[0] for name in self.activation_names])
        
        readable_metadata = f"{self.network_type.__name__}-{plus_extra}"
        
        return "exp-" + readable_metadata + "-" + hash_monstrosity[:maxlen] 
    
    def exp_vis_params(self) :
        
        """
        Generates the visual parameters dataclass for this activation function. 

        Returns:
            The visual parameters dataclass, with its associated attributes and methods.
        """
        
        main_params = {
            "save_folder" : self.savename(),
            "activation_colours" : dict(zip(self.activation_names, 
                                            get_n_colours(len(self.activations)) )),
            "kf_aggfunc_linestyles" : dict(zip(self.kfold_aggfunc_names, 
                                               list(mlines.lineStyles.keys())[:len(self.kfold_aggfuncs)])),
            "experiment" : self,
        }
        
        return expVisParams(**main_params)
    
@dataclass
class expVisParams() :
    
    """
    This is the visual parameters dataclass for a given experiment. 
    
    Params:
        save_folder: save folder name for the experiment, generated by experiment.savename().
        activation_colours: different colours for each activation function for differentiation.
        kf_aggfunc_linestyles: different linestyles per kfold aggregation function for differentiation.
        experiment: reference to the original experiment object that created it.
        
    """
    
    save_folder : str
    activation_colours : dict[str, Any]
    kf_aggfunc_linestyles : dict[str, str]
    experiment : experimentParams
    
    def initialise_figure_params(self) -> dict[tuple[str, str, str], Any] :
        
        """Creates default parameters dictionary for all valid figures.

        Returns:
            A dictionary of triples, where each triple corresponds to an
            (eval_type (train/test), category, measure_type) combination.
            Each associated value is itself the dictionary of figure parameters for that combination.
        """
        
        figures_dict = {}
        
        for eval_type in ("train", "test") : 
            for category in self.experiment.categories :
                for measure_type in category2measure_types(category) :
                    combination = (eval_type, category, measure_type)
                    figures_dict[combination] = self.generate_figure_params(*combination)
    
        return figures_dict
    
    def generate_figure_params(self, eval_type : str, category : str, measure_type : str,
                               plots_per_row : int = 3) -> dict[str, Any] :
        
        """
        Generate parameters dictionary for a given figure and triple (eval_type, category, measure_type).
        
        Params:
            eval_type: the evaluation type (train/test) of the data. 
            category: what type of data (gradient, testloss, test predictions, etc) is being measured.
            measure_type: what the valid dimension (independent variable) is. Usually "epochs", but not always.
            plots_per_row: how many plots should be represented on each given row. Defaults to 3 for appearances.

        Returns:
            dictionary of parameters for the given figure, to be plugged in during visualisation.
        """
        
        nrows = ceil(len(self.experiment.test_functions) / plots_per_row)
        ncols = min(plots_per_row, len(self.experiment.test_functions))
        
        # For testloss there are no tests we can perform, so will always be exactly 1 figure even if other tests exist
        special_cases = { "testloss" : (1, 1) }
        if category in special_cases : nrows, ncols = special_cases[category]
        
        title = generate_plot_title(category, 1 if eval_type == "test" else self.experiment.kfold_k)
        plot_type = determine_plot_type(eval_type, category, measure_type)
        
        # All curves get symlog treatment for numerical stability
        if plot_type == "curve" : title += " (symlog)"
        
        fig_params = {
            "savename" : f"{category}-{measure_type}_on_{eval_type}",
            "figsize" : (1920/96, 1080/96),
            "eval_type" : eval_type,
            "category" : category,
            "measure_type" : measure_type,
            "title" :  title,
            "plot_type" : plot_type,
            "nrows" : nrows,
            "ncols" :  ncols,
            "nplots" : nrows * ncols           
        }
        
        return fig_params
    
    def generate_axes_params(self, test_type : str, fig_params : dict[str, Any], nskip: int = 5) -> dict[str, Any] :
        
        """Generates parameters for the given axes object on a given figure.
        
        Params:
            test_type: what test function the axes is for. Every axes object is for a specific test function.
            
            fig_params: the associated dictionary of parameters for the parent figure object. Use generate_figure_params()
            if this is not available from the same dataclass object expVisParams.
            
            nskip: Number of initial epochs to skip. Only valid for epochs or ordered x-axes. 

        Returns:
            Dictionary of parameters for the given axes object.
        """
        
        match fig_params["plot_type"] :
            case "curve" :  # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"
                xlabel, ylabel = (fig_params["measure_type"][:-1], test_type) 
            case "kde" | "histplot" :
                xlabel, ylabel = (test_type, "frequency density")
            case _ :
                xlabel, ylabel = ("x-axis placeholder label", "y-axis placeholder label")

        xticklabels = sampling_indices(self.experiment.epochs, self.experiment.max_samples)
        
        ax_params = { 
            "test_type" : test_type, 
            "plot_type" : fig_params["plot_type"],
            "xlabel" : xlabel,
            "ylabel" : ylabel,
            "grid" : True, 
            "nxticks" : 10,
            "xticklabels" : xticklabels,
            "nskip" : nskip
        }
    
        return ax_params

    def generate_plot_params(self, activation_name : str, agg_type : str, plot_type : str) :
        
        """
        Generates parameters for the given plot object for an axes object (axes itself not required).
        
        Params:
            activation_name: the name of the activation to plot over (determines colour).
            agg_type: the type of aggregation function used (determines linestyle).
            plot_type: the type of plot (kde, curve, histplot, etc).

        Returns:
            The dictionary of associated parameters for the axes object.
        """
        
        plot_params = {
            "activation_name" : activation_name,
            "agg_type" : agg_type,
            "plot_type" : plot_type,
            "label" : f"fold-{agg_type}({activation_name})",
            "colour" : self.activation_colours[activation_name],
            "linestyle" : self.kf_aggfunc_linestyles[agg_type],
            "markersize" : 4.0,
            "marker" : "^"
        }
        
        return plot_params

@dataclass
class categoryParams() :
    name : str
    tester : Callable = dummy_idfunc
    measure_types : tuple[str, ...] = ()
    
    def __post_init__(self) :
        
        self.measure_types = category2measure_types(self.name)
        
        # I hate that this is necessary, but there is no other way to get this to work without fusing .py files
        if self.tester.__name__ == "dummy_idfunc":
            import activation_testing
            self.tester = getattr(activation_testing, "post_experiment_test_" + self.name)
    
@dataclass
class monitorParams() :
    X : torch.Tensor
    test_functions : tuple[Callable, ...]
    test_function_names : str | list[str]
    kfold_aggfuncs :  tuple[Callable, ...]
    kfold_columns : str | list[str]
    
    def validate(self, expected_ndims : int = 2) :
        
        if not isinstance(self.test_function_names, list) : self.test_function_names = [self.test_function_names]
        if not isinstance(self.kfold_aggfuncs, tuple) : self.kfold_aggfuncs = ( self.kfold_aggfuncs, )
        if not isinstance(self.kfold_columns, list) : self.kfold_columns = [self.kfold_columns]
        
        is_test_data = len(self.X.size()) == expected_ndims
        
        # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
        if is_test_data : self.X = self.X[..., None]

        self.test_function_names = validate_activation_df_column_names(self.test_functions, self.test_function_names)
        self.kfold_columns = validate_activation_df_column_names(self.kfold_aggfuncs, self.kfold_columns)

@dataclass
class experimentResult() :
    
    """Simple container class for efficiently representing all categories of result from an experiment. 
        Not intended for any complex calculations, unlike experimentParams or expVisParams.
    """
    
    grad : torch.Tensor
    testloss : torch.Tensor
    testpreds : torch.Tensor

    def get_ndims(self) -> dict[str, int] :
        
        ndims = { name : len(x.size()) for name, x in self.__dict__.items() if isinstance(x, torch.Tensor)}
        
        return ndims
    
    def get_ndims_tuple(self) -> tuple :
        
        ndims = tuple( ( name , len(x.size()) ) for name, x in self.__dict__.items() if isinstance(x, torch.Tensor) )
        
        return ndims
    
    def get_max_dim(self) -> int :
        
        return max(self.get_ndims_tuple())



