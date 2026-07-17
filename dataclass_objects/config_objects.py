from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from typing import Any, Callable
import torch
from torch import nn
from torch.nn import Identity
import matplotlib.lines as mlines
import json
import hashlib
from math import ceil
import numpy as np

# CUSTOM
from networks import ActivationNetwork
from support.parsing_helpers import validate_activation_df_column_names, is_hashable, smart_str, safe_asdict, get_name, extract_tuple_list
from support.processing_helpers import sampling_indices, params2grad_vector, pad_torch_stack
from support.plotting_helpers import get_n_colours, determine_plot_type, category2name
from support.torch_reducers import arithmetic_mean


@dataclass
class expConfig() :
    
    """
    Main dataclass for conducting activation experiments efficiently. This dataclass is designed
    to compare the effectiveness of different activation functions on a given network using a set of 
    test functions (reducers) to marginalise unwanted dimensions and compare 1D arrays of results.
    
    
    Params:
        df_train: dataframe for train data.
        df_test: dataframe for test data.
        labels: list of columns that are to be predicted from features (presumed rest)
        network_type: type of neural network to work with for the experiment and perform predictions.
        loss: loss metric to judge predictions by. 
        feature_transforms: tuple of tuples, where the first entry of each tuple is the list of columns to be transformed,
        and the second entry is the transformer class to apply the transformation to each of the columns.
        label_transforms: same as feature_transforms, but for labels.
        reducers: tuple of test functions to apply on finished results to marginalise unwanted dimensions. 
        activations: tuple of all activation functions to run in the experiment.
        kf_reducers: tuple of all aggregation functions to collapse folds over in K-fold crossvalidation, 
        e.g mean, variance, etc over folds. Note for test data it's interpreted as 1-fold crossvalidation, and only 
        mean is permitted for this.
        lr: constant learning rate. Can be changed to a variable one.
        kfold_k: number of folds to use in Kfold cross validation. 10 is the industry standard.
        n_alphas: number of different alpha values in the range (0,1) to use for LS sensitivity testing, with uniform spacing.
        E.g n_alphas = 5 would imply the alpha values (0.0, 0.25, 0.50, 0.75, 1.0).
        batch_size: size of train set used per iteration per epoch. Higher values increase gradient accuracy at the cost of time.
        max_recorded_samples: maximum number of samples to record (does not change number of epochs). Higher values create denser graphs.
        epochs: number of total training runs to apply to the neural network, where each epoch is a full pass of df_train.
        activation_names: list of display names for the activation functions, index-linked with activations.
        reducer_names: list of display names for the test functions. Again, index-linked with reducers.
        kf_reducer_names: list of display names for the aggregation functions, index-linked with kf_reducers.
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
    activations : list[nn.Module] | tuple[nn.Module, ...]
    reducers : tuple[Callable, ...]
    kf_reducers : tuple[Callable, ...]
    lr : float = 0.001
    kfold_k : int = 10
    n_alphas : int = 5
    batch_size : int = -1 # full batch
    max_recorded_samples : int = -1 # no limit
    epochs : int = 500
    activation_names : list[str] = field(default_factory = list)
    reducer_names : list[str] = field(default_factory = list)
    kf_reducer_names : list[str] = field(default_factory = list)
    features_dtype : torch.dtype = torch.float32
    labels_dtype : torch.dtype = torch.long
    categories : tuple[str, ...] | str = ("grad",)
    
    def __post_init__(self) :
        
        """
        Validation of all inputs to make sure the experiment runs smoothly and every activation has a valid name.
        """
        
        # YAML doesn't support tuples so convert in the program
        for potentially_list_attribute in ( "activations", "reducers", "kf_reducers", "categories" ) :
            attr = getattr(self, potentially_list_attribute)
            if isinstance(attr, list) :
                setattr(self, potentially_list_attribute, tuple(attr))
        
        # Placate the linter + consistency
        if isinstance(self.labels, str) : 
            self.labels = [self.labels]
        
        # Same as above
        if isinstance(self.categories, str) :
            self.categories = (self.categories, )

        self.reducer_names = validate_activation_df_column_names(self.reducers, self.reducer_names)
        self.activation_names = validate_activation_df_column_names(self.activations, self.activation_names)
        self.kf_reducer_names = validate_activation_df_column_names(self.kf_reducers, self.kf_reducer_names)
        
        # Mean is non-optional for expConfig, we need it for when we collect results over test data
        if "arithmetic_mean" not in {get_name(f) for f in self.kf_reducers } : # Use this since funcs have no ==
            self.kf_reducer_names.append("mean")
            self.kf_reducers += (arithmetic_mean, )
        
        # Used so we can access ._structure, .height and .width easily
        self.dummy = self.network_type(Identity)
                
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
        strformat = str([ smart_str(v) if is_hashable(v) else smart_str(get_name(v))
                         for v in self.__dict__.values() ])
        as_str = json.dumps(strformat + "you and I, (nothing comes easy) * 2", sort_keys = True).encode() 
        
        hash_monstrosity = hashlib.sha256(as_str).hexdigest()
        
        if len(self.activation_names) == 1 :
            plus_extra = f"{get_name(self.activations[0].base_activation)}_alpha[{self.n_alphas}]"
        else :
            plus_extra = "_".join([name[0] for name in self.activation_names])
        
        readable_metadata = f"{self.network_type.__name__}-{plus_extra}"
        
        return "experiments/exp-" + readable_metadata + "-" + hash_monstrosity[:maxlen] 
    
    def exp_vis_params(self) -> expVisual :
        
        """
        Generates the visual parameters dataclass for this activation function. 

        Returns:
            The visual parameters dataclass, with its associated attributes and methods.
        """
        
        possible_linestyles = ["-", "--", ] +  list(mlines.lineStyles.keys())
        possible_markers = [".", "^",] + list(mlines.lineMarkers.keys())
        
        main_params = {
            "save_folder" : self.savename(),
            "activation_colours" : dict(zip(self.activation_names, get_n_colours(len(self.activations)) )),
            "category_linestyles" :  dict(zip(list(self.categories), possible_linestyles[:len(self.categories)])),
            "kf_aggfunc_markerstyles" : dict(zip(self.kf_reducer_names, possible_markers[:len(self.kf_reducers)])),
            "experiment" : self,
        }
        
        return expVisual(**main_params)
    

    
@dataclass
class expVisual() :
    
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
    category_linestyles : dict[str, str]
    kf_aggfunc_markerstyles : dict[str, str]
    experiment : expConfig
    
    def generate_figure_title(self, eval_type : str, category : str, measure_type : str ) -> str :
        
        fold_explanation = f", {self.experiment.kfold_k}-fold" if eval_type == "train" else ""
        title = f"{category2name(category)} {eval_type} data{fold_explanation} results measured over {measure_type}"
        
        return title
    
    def generate_figure_params(self, eval_type : str, category : str, measure_type : str,
                               plots_per_row : int = 2) -> dict[str, Any] :
        
        """
        Generate parameters dictionary for a given figure and triple (eval_type, category, measure_type).
        
        Params:
            eval_type: the evaluation type (train/test) of the data. 
            category: what type of data (gradient, testloss, test predictions, etc) is being measured.
            measure_type: what the valid dimension (independent variable) is. Usually "epochs", but not always.
            plots_per_row: how many plots should be represented on each given row. Defaults to 2 for appearances.

        Returns:
            dictionary of parameters for the given figure, to be plugged in during visualisation.
        """
        
        nrows = ceil(len(self.experiment.reducers) / plots_per_row)
        ncols = min(plots_per_row, len(self.experiment.reducers))
        
        # For testloss there are no tests we can perform, so will always be exactly 1 figure even if other tests exist
        special_cases = { "testloss" : (1, 1) }
        if category in special_cases : nrows, ncols = special_cases[category]
        
        title = self.generate_figure_title(eval_type, category, measure_type)
        plot_type = determine_plot_type(eval_type, category, measure_type)
        
        # All curves get symlog treatment for numerical stability
        if plot_type == "curve" and category != "metrics" : title += " (symlog)"
        
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
    
    def generate_xaxis(self, measure_type : str) -> list[int] :
    
        match measure_type :
            case "epochs" :
                return sampling_indices(self.experiment.epochs, self.experiment.max_recorded_samples)
            case "layers" :
                return list(range(self.experiment.dummy.length))
            case _ :
                return []
    
    def generate_axes_params(self, reducer : str, fig_params : dict[str, Any], nskip: int = 5) -> dict[str, Any] :
        
        """Generates parameters for the given axes object on a given figure.
        
        Params:
            reducer: what test function the axes is for. Every axes object is for a specific test function.
            
            fig_params: the associated dictionary of parameters for the parent figure object. Use generate_figure_params()
            if this is not available from the same dataclass object expVisual.
            
            nskip: Number of initial epochs to skip. Only valid for epochs or ordered x-axes. 

        Returns:
            Dictionary of parameters for the given axes object.
        """
        
        match fig_params["plot_type"] :
            case "curve" :  # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"
                xlabel, ylabel = (fig_params["measure_type"][:-1], reducer) 
            case "kde" | "histplot" :
                xlabel, ylabel = (reducer, "frequency density")
            case _ :
                xlabel, ylabel = ("x-axis placeholder label", "y-axis placeholder label")

        ax_params = { 
            "reducer" : reducer, 
            "plot_type" : fig_params["plot_type"],
            "xlabel" : xlabel,
            "ylabel" : ylabel,
            "grid" : True, 
            "nxticks" : 10,
            "xaxis" : self.generate_xaxis(fig_params["measure_type"]),
            "nskip" : nskip
        }
    
        return ax_params

    def generate_plot_params(self, activation_name : str, category : str, kf_reducer : str, plot_type : str) :
        
        """
        Generates parameters for the given plot object for an axes object (axes itself not required).
        
        Params:
            activation_name: the name of the activation to plot over (determines colour).
            kf_reducer: the type of aggregation function used (determines linestyle).
            plot_type: the type of plot (kde, curve, histplot, etc).

        Returns:
            The dictionary of associated parameters for the axes object.
        """
        
        plot_params = {
            "activation_name" : activation_name,
            "kf_reducer" : kf_reducer,
            "plot_type" : plot_type,
            "label" : f"fold-{kf_reducer}({activation_name})",
            "colour" : self.activation_colours[activation_name],
            "linestyle" : self.category_linestyles[category],
            "marker" : self.kf_aggfunc_markerstyles[kf_reducer],
            "markersize" : 5.0,
        }
        
        return plot_params