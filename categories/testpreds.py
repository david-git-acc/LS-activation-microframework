from __future__ import annotations
from dataclasses import replace
import torch
import pandas as pd

### CUSTOM
from dataclass_objects.input_objects import expInput, testInput
from categories.base_definitions import categoryExperimentTracker
from categories.grad import post_experiment_test_grad



class testpredsExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for test prediction data. Stores epochs and test samples.
    Note that NaN values will be used as padding whenever stratified K-fold differs in size. As long as using
    a NaN-aware metric e.g those found in helpers (arithmetic_mean, variance, log_average), this will not impact the results.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "testpreds"
        self._data = torch.full(size = (xpi.n_captures, len(xpi.Y_test_tensor)), fill_value = torch.nan) 
    
    def track(self) -> torch.Tensor :
        return torch.argmax(self.xpi.anet_model(self.xpi.X_test_tensor), dim = 1).view(-1).detach().cpu()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()  


def post_experiment_test_testpreds(ti : testInput) -> pd.DataFrame :
    
    """
    Same as post_experiment_test_grad, but for test predictions (testpreds). Identical logic.
    
    Returns:
        result_df: Pandas dataframe containing results. Each column is a different agg-type test-type combination.
    """
        
    return post_experiment_test_grad(replace(ti, expected_ndims = 2 ))