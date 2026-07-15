from __future__ import annotations
from dataclasses import replace
import torch
import pandas as pd

### CUSTOM
from dataclass_objects.input_objects import expInput, testInput
from support.config import config
from support.torch_reducers import donothing_dummy
from categories.base_definitions import categoryExperimentTracker
from categories.grad import post_experiment_test_grad



class testlossExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for testloss data. Stores epochs only by default.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "testloss"
        self._data = torch.full(size = (xpi.n_captures,), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return self.xpi.target_loss(self.xpi.anet_model(self.xpi.X_test_tensor), self.xpi.Y_test_tensor).detach().cpu().item()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index] = self.track()
        

def post_experiment_test_testloss(ti : testInput) -> pd.DataFrame :
    
    """
    Same as post_experiment_test_grad, but for testloss. Identical logic.
    Note that test_suite, over and test_columns are deprecated because only 1 dimension is supported.

    Returns:
        results_df: dataframe of results. Each column is a different agg-type.
    """
    
    return post_experiment_test_grad(replace(ti, reducers = (donothing_dummy, ), reducer_names = ["test loss"], 
                                             expected_ndims = 1, measure_type = "epochs"))