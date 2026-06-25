from dataclasses import dataclass, asdict, fields, field
import pandas as pd
from networks import ActivationNetwork
from typing import Any, Callable
import torch
from torch import nn
from helpers import validate_activation_df_column_names, get_name


@dataclass
class experimentParams() :
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    labels : str | list[str]
    network_type : type[nn.Module]
    feature_transforms : tuple[tuple[list[str], Any]]
    label_transforms : tuple[tuple[list[str], Any]]
    test_suite : tuple[Callable, ...]
    activations : tuple[nn.Module, ...]
    kfold_aggfuncs : tuple[Callable, ...]
    kfold_k : int = 10
    activation_names : list[str] = field(default_factory = list)
    kfold_aggfunc_names : list[str] = field(default_factory = list)
    
    def __post_init__(self) :
        
        # Placate the linter + consistency
        if isinstance(self.labels, str) : 
            self.labels = [self.labels]

        self.activation_names = validate_activation_df_column_names(self.activations, self.activation_names)
        self.kfold_aggfunc_names = validate_activation_df_column_names(self.kfold_aggfuncs, self.kfold_aggfunc_names)
    
        
@dataclass
class categoryParams() :
    name : str
    tester : Callable
    measure_types : tuple[str, ...]
    
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
        
        ndims = { x.__name__ : len(x.size()) for x in fields(self) if isinstance(x, torch.Tensor)}
        
        return ndims
    
    def get_ndims_tuple(self) -> tuple :
        
        ndims = tuple( len(x.size()) for x in fields(self) if isinstance(x, torch.Tensor) )
        
        return ndims
    
    def get_max_dim(self) -> int :
        
        return max(self.get_ndims_tuple())
    
@dataclass
class plotParams() :
    plot_name : str = ""
    plot_type : str = "curve"
    colour : str = "gold"
    linestyle : str = "--"
    marker : str = "^"
    markersize : int = 4
    xlabel : str = "x-axis"
    ylabel : str = "y-axis"
    legend_label : str = "plot"
    
