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


# This is for grad, but it generalises well, so you can use this as a base function to define post_experiment_test on
def post_experiment_test_grad(ti : testInput) -> pd.DataFrame :
    
    """
    Perform the function test suite on a designated set of test functions with k-folds, then collapses over the k-folds using
    an aggregation function (typically mean) and returns results as a Pandas dataframe.
    
    Params:
        ti: testInput dataclass. See the type annotation for more information on required parameters for it. 
        
    NOTE: This function is the main function for post experiment testing; testloss and testpreds rely on this one. Also,
    remember that the last dimension must always be the kfold dimension, or bugs will occur - silently or not.

    Returns:
        result_df: Pandas dataframe containing results. Each column is a different agg-type test-type combination.
    """
    
    # The dimensions to marginalise in are always all dimensions except the dimension we care about 
    # + the kfold dimension (last)
    dim = tuple(i for i in range(len(ti.X.size()) - 1 ) if i != name2index(ti.measure_type) ) 
    
    # Store everything we collect here
    test_results = []
    
    # Store the column names in a given format so easier to store
    df_columns = []
    
    # For each test function we compute the result and collapse all other dimensions using it, to get a 2D array (dim, folds)
    for i, test_func in enumerate( ti.reducers ) : 
        result = test_func(ti.X, dim = dim) # Make sure all test suite functions is NaN-aware + can handle any subset of dims
        
        # Then we collapse the fold dimension in different ways; these are important, will be covered later.
        for j, kfold_aggfunc in enumerate( ti.kf_reducers ) :
        
            kfold_dim = len(result.size()) - 1
        
            collapsed_result = kfold_aggfunc(result, dim = kfold_dim )
            data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
            test_results.append(data)
            
            df_column_name = (ti.reducer_names[i], ti.kf_reducer_names[j])
            df_columns.append(df_column_name)
    
    assert len(set(df_columns)) == len(df_columns), f"Duplicate df columns. Please check testfuncs and kf_reducers"
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    result_df = pd.DataFrame(test_results, columns = df_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = ti.measure_type[:-1] # Kill the "s", we view singularly
        
    return result_df    


class gradExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for gradient data. Stores epochs and parameters by default.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "grad"
        self._data = torch.full(size = (xpi.n_captures, *self.xpi.nabla_shape), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return params2grad_vector(self.xpi.anet_model.parameters())
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()