from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable
from abc import ABC, abstractmethod
import torch
import pandas as pd
import numpy as np 

### CUSTOM
from dataclass_objects import expInput, experimentResult, testInput
from support.config import config
from support.processing_helpers import params2grad_vector, dfs_settings2tensors
from support.parsing_helpers import name2index, safe_asdict
from support.torch_reducers import donothing_dummy
from categories.base_definitions import categoryExperimentTracker


class metricsExperimentTracker(categoryExperimentTracker) :
    
    """metricsExperimentTracker implementation for test prediction data. Stores epochs and test samples.
    This implementation is currently identical to testpredsExperimentTracker except the _category = "metrics".
    However, it is kept separately to avoid unnecessary dependency and allow for future isolation of any changes.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "metrics"
        self._data = torch.full(size = (xpi.n_captures, len(xpi.Y_test_tensor)), fill_value = torch.nan) 
    
    def track(self) -> torch.Tensor :
        return torch.argmax(self.xpi.anet_model(self.xpi.X_test_tensor), dim = 1).view(-1).detach().cpu()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()  


def post_experiment_test_metrics(ti : testInput) -> pd.DataFrame :
    
    """Post-experiment test function for metrics. 
    
    Params:
        testInput dataclass, containing the dataframes, reference to the experiment configuration (may be None),
        and all necessary parameters. See the TestInput type annotation for more details.
    
    Returns:
        DataFrame: DataFrame containing all (metric, kf_reducer) combinations and the corresponding output data.
    
    """
    
    # We need an experiment reference so we can identify which data to compare to
    if ti.xpc is None :
        raise ValueError("Metrics category requires a parent expConfig reference (xpc) in testInput dataclass")
    
    kfold_data = ti.metadata.get("kfold_data", None)
    if kfold_data is None : # If it's test data, then we need to use test data as data source
        processed_kf_data = [dfs_settings2tensors(**safe_asdict(ti.xpc, dfs_settings2tensors), 
                                              dtypes = (ti.xpc.features_dtype, ti.xpc.labels_dtype))] 
    else :
        # We have to use the train and test data we got, same as before.
        processed_kf_data = [dfs_settings2tensors(train_df, test_df, ti.xpc.feature_transforms, ti.xpc.label_transforms,
                                                  ti.xpc.labels, (ti.xpc.features_dtype, ti.xpc.labels_dtype)) 
                                                  for train_df, test_df in kfold_data]
    
    # Ground truths are Y test, which will be the 4th and final element of the processing
    ground_truths = [Y_test.numpy() for X_train, X_test, Y_train, Y_test in processed_kf_data ]
    
    nepochs = ti.X.shape[name2index("epochs")]
    nfolds = ti.X.shape[-1]
    
    result_dict = { metric_name : torch.full(size = (nepochs, nfolds), fill_value = torch.nan) 
                   for metric_name in config["eval_metric_names"] }
    
    for metric_name, metric in zip(config["eval_metric_names"], config["eval_metrics"]) :
        
        # The ground truths only change by fold, never by epochs or samples, so we iterate only over folds
        for k in range(nfolds) :
            
            # 1D array containing true sizes, without padding of the correct values for given testpreds
            y = ground_truths[k]
            testpreds_this_fold = ti.X[:, :, k] # Must be unpadded
            
            # The NaN-padding is always the same across all epochs, so we just check the first
            without_nanpadding = ~torch.isnan(testpreds_this_fold[0, :])   
            unpadded_testpreds = testpreds_this_fold[:, without_nanpadding].numpy()
            
            # Forced to iterate manually because metrics are not vectorisable
            result = torch.tensor([metric(y, y_hat) for y_hat in unpadded_testpreds])
            result_dict[metric_name][:, k] = result
          
        for kfold_aggfunc_name, kfold_aggfunc in zip(ti.kf_reducer_names, ti.kf_reducers ) :
            result = result_dict[metric_name]
            collapsed_result = kfold_aggfunc(result, dim = 1 ) # Always 2D so last dim is always = 1
            result_dict[(metric_name, kfold_aggfunc_name)] = collapsed_result.view(-1).numpy()
        
        # Once we have applied all agg funcs over all folds, no longer need the original uncollapsed kfold data
        del result_dict[metric_name]
            
    result_df = pd.DataFrame(result_dict)
    result_df.columns = pd.Index(result_df.columns) # Multiindex will break the code later
    
    if (diff := len(set(result_df.columns)) - len(result_df.columns)) != 0 :    
        raise ValueError(f"{diff} duplicate df columns detected. Please check eval_metrics and kf_reducers")
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = "epoch" 
        
    return result_df    