from dataclasses import dataclass, asdict, fields, field
import pandas as pd
from networks import ActivationNetwork
from typing import Any, Callable
import torch
from torch import nn
from helpers import (validate_activation_df_column_names, sampling_indices, get_n_colours, dummy_idfunc, 
                     category2measure_types, determine_plot_type, generate_plot_title, safe_set_params, arithmetic_mean,
                     instantiate_default_params, is_hashable, smart_str)
import matplotlib.lines as mlines
import json
import hashlib
from math import ceil
from itertools import product


@dataclass
class experimentParams() :
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    labels : str | list[str]
    network_type : type[nn.Module]
    loss : nn.Module
    feature_transforms : tuple[tuple[list[str], Any]]
    label_transforms : tuple[tuple[list[str], Any]]
    test_suite : tuple[Callable, ...]
    activations : tuple[nn.Module, ...]
    kfold_aggfuncs : tuple[Callable, ...]
    lr : float = 0.001
    kfold_k : int = 10
    n_alphas : int = 5
    batch_size : int = -1 # full batch
    max_samples : int = -1 # no limit
    epochs : int = 500
    activation_names : list[str] = field(default_factory = list)
    test_columns : list[str] = field(default_factory = list)
    kfold_aggfunc_names : list[str] = field(default_factory = list)
    features_dtype : torch.dtype = torch.float32
    labels_dtype : torch.dtype = torch.long
    categories : tuple[str, ...] = ("grad", "testloss", "testpreds")
    
    def __post_init__(self) :
        
        # Placate the linter + consistency
        if isinstance(self.labels, str) : 
            self.labels = [self.labels]

        self.test_columns = validate_activation_df_column_names(self.test_suite, self.test_columns)
        self.activation_names = validate_activation_df_column_names(self.activations, self.activation_names)
        self.kfold_aggfunc_names = validate_activation_df_column_names(self.kfold_aggfuncs, self.kfold_aggfunc_names)
        
        # Mean is non-optional for experimentParams, we need it for when we collect results over test data
        if arithmetic_mean not in self.kfold_aggfuncs :
            self.kfold_aggfunc_names.append("mean")
            self.kfold_aggfuncs += (arithmetic_mean, )
    
    def savename(self , maxlen : int = 10) -> str :
        
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
    save_folder : str
    activation_colours : dict[str, Any]
    kf_aggfunc_linestyles : dict[str, str]
    experiment : experimentParams
    
    def initialise_figure_params(self) -> dict[tuple[str, str, str], Any] :
        
        figures_dict = {}
        
        for eval_type in ("train", "test") : 
            for category in self.experiment.categories :
                for measure_type in category2measure_types(category) :
                    combination = (eval_type, category, measure_type)
                    figures_dict[combination] = self.generate_figure_params(*combination)
    
        return figures_dict
    
    def generate_figure_params(self, eval_type : str, category : str, measure_type : str,
                               plots_per_row : int = 3) -> dict[str, Any] :
        
        nrows = ceil(len(self.experiment.test_suite) / plots_per_row)
        ncols = min(plots_per_row, len(self.experiment.test_suite))
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
        
        if fig_params["plot_type"] == "curve" :
            # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"
            xlabel, ylabel = (fig_params["measure_type"][:-1], test_type) 
        elif fig_params["plot_type"] in ["kde", "histplot"] :
            xlabel, ylabel = (test_type, "frequency density")
        else : 
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
    test_suite : tuple[Callable, ...]
    test_columns : str | list[str]
    kfold_aggfuncs :  tuple[Callable, ...]
    kfold_columns : str | list[str]
    
    def validate(self, expected_ndims : int = 2) :
        
        if not isinstance(self.test_columns, list) : self.test_columns = [self.test_columns]
        if not isinstance(self.kfold_aggfuncs, tuple) : self.kfold_aggfuncs = ( self.kfold_aggfuncs, )
        if not isinstance(self.kfold_columns, list) : self.kfold_columns = [self.kfold_columns]
        
        is_test_data = len(self.X.size()) == expected_ndims
        
        # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
        if is_test_data : self.X = self.X[..., None]

        self.test_columns = validate_activation_df_column_names(self.test_suite, self.test_columns)
        self.kfold_columns = validate_activation_df_column_names(self.kfold_aggfuncs, self.kfold_columns)

@dataclass
class experimentResult() :
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



