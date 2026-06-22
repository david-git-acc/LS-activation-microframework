from dataclasses import dataclass, asdict, fields
import pandas as pd
from networks import ActivationNetwork
from typing import Any, Callable
import torch
from torch import nn
from helpers import validate_activation_df_column_names


@dataclass
class experimentParams() :
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    labels : str | list[str]
    network_type : type[nn.Module]
    feature_transform_list : list[tuple[list[str], Any]]
    label_transform_list : list[tuple[list[str], Any]]
    test_suite : list[nn.Module]
    activations : list[nn.Module]
    kfold_aggfuncs : list[nn.Module]
    kfold_k : int = 10
    
        
@dataclass
class categoryParams() :
    name : str
    tester : Callable
    measure_types : list[str]
    
@dataclass
class monitorParams() :
    X : torch.Tensor
    test_suite : list[Callable]
    test_columns : str | list[str]
    kfold_aggfuncs :  list[Callable]
    kfold_columns : str | list[str]
    
    def validate(self, expected_ndims : int = 2) :
        
        if not isinstance(self.kfold_aggfuncs, list) : self.kfold_aggfuncs = [self.kfold_aggfuncs]
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