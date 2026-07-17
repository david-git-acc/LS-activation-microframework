from __future__ import annotations
import torch
import pandas as pd
import numpy as np 
from dataclasses import replace

### CUSTOM
from support.processing_helpers import pad_torch_stack
from categories.base_definitions import categoryExperimentTracker
from categories.grad import post_experiment_test_grad
from dataclass_objects.input_objects import expInput, testInput

class aoutsExperimentTracker(categoryExperimentTracker) :
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        
        self._category = "aouts"
        activation_data_shape = (self.xpi.anet_model.n_activations, self.xpi.anet_model.width)
        self._data = torch.full(size = (self.xpi.n_captures, *activation_data_shape), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        
        outs_this_epoch = self.xpi.anet_model.activation_outs
        
        # Marginalise over batch size, we're not interested in tracking this; at most per neuron
        mean_outs_this_epoch = [out.nanmean(dim = 0).detach().cpu() for out in outs_this_epoch.values()]
        padded_outs = pad_torch_stack(mean_outs_this_epoch, pad_with = torch.nan)
        
        # List dimension is over layers, so need dim = 0
        stacked_outs = torch.stack(padded_outs, dim = 0)
        
        return stacked_outs
    
    def record(self, record_index : int = 0) -> None :
        
        self._data[record_index, :, :] = self.track()
        

def post_experiment_test_aouts(ti : testInput) -> pd.DataFrame :
    
    return post_experiment_test_grad(replace(ti, expected_ndims = 3))