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

class agradExperimentTracker(categoryExperimentTracker) :
    def __init__(self, xpi : expInput ) :
        super().__init__(xpi)
        
        self._category = "agrad"
        activation_data_shape = (self.xpi.anet_model.length, self.xpi.anet_model.width)
        self._data = torch.full(size = (self.xpi.n_captures, *activation_data_shape), fill_value = torch.nan)

        
    def track(self) -> torch.Tensor :
        
        grads_this_epoch = self.xpi.anet_model.activation_grads 
        
        for layer_index, grad in enumerate( grads_this_epoch ) :
            if grad.numel() == 0 :
                raise ValueError(f"Layer {layer_index} of activation grad data not initialised")
        
        # It's backpropagation so we will always get them back-to-front
        grads_this_epoch.reverse()
        
        # Marginalise over batch size, we're not interested in tracking this; at most per neuron
        mean_grads_this_epoch = [grad.nanmean(dim = 0).detach().cpu() for grad in grads_this_epoch]
        
        padded_grads = pad_torch_stack(mean_grads_this_epoch, pad_with = torch.nan)
        
        # List dimension is over layers, so need dim = 0
        stacked_grads = torch.stack(padded_grads, dim = 0)
        
        return stacked_grads
     
    def record(self, record_index : int = 0) -> None :
        
        self._data[record_index, :, :] = self.track()
        

def post_experiment_test_agrad(ti : testInput) -> pd.DataFrame :
    
    return post_experiment_test_grad(replace(ti, expected_ndims = 3))