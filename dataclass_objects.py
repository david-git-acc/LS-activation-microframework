from dataclasses import dataclass, asdict
import pandas as pd
from networks import ActivationNetwork
from typing import Any, Callable
import torch
from torch import nn


@dataclass
class experimentParams() :
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    target_columns : str | list[str]
    network_type : type[nn.Module]
    feature_transform_list : list[tuple[list[str], Any]]
    label_transform_list : list[tuple[list[str], Any]]
    test_suite : list[nn.Module]
    activations : list[nn.Module]
    kfold_aggfuncs : list[nn.Module]
    
    
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
    kfold_aggfuncs : Callable | list[Callable]
    kfold_columns : str | list[str]
    