from __future__ import annotations
import torch
import pandas as pd
from dataclasses import replace

### CUSTOM
from dataclass_objects.input_objects import expInput, testInput
from support.config import config
from categories.grad import post_experiment_test_grad
from categories.base_definitions import categoryExperimentTracker

class lsExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for LS-alpha data; tracking the alpha values per layer of the network 
    per epoch. If applied to a non-LS activation, will return an array of NaNs. If applied to an LS activation with
    LS parameter learnable = False, will return a constant array equal to LS.alpha * torch.ones. 
    """
    
    def __init__(self, xpi : expInput) -> None :
        super().__init__(xpi)
        self._category = "ls" # Of size (n_epochs, n_activation_layers)
        self._data = torch.full(size = (xpi.n_captures, self.xpi.anet_model.n_activations), fill_value = torch.nan)
        self.warned = False # Warn if we encounter NaN values in the layers
        
    def track(self) -> torch.Tensor :
        
        # Have to consider possibility some activations are LS and some are not for generality, hasattr is O(1) check
        layer_alphas = torch.tensor([activ.alpha.detach().cpu() # Placate linter, thinks alpha is not a tensor
                                     if hasattr(activ, "alpha") and isinstance(activ.alpha, torch.Tensor) else torch.nan
                                     for _, activ in self.xpi.anet_model.activation_layers ], dtype = torch.float32)
        
        if not self.warned and torch.isnan(layer_alphas).any().item() : 
            print("Warning: NaN values detected in LS-alpha tracking. Non-LS activations may exist inside the network!")
            self.warned = True
        
        return layer_alphas
    
    def record(self, record_index : int = 0) -> None :
        
        self._data[record_index, :] = self.track()

def post_experiment_test_ls(ti : testInput) -> pd.DataFrame :
    
    """Post-experiment test function for LS alpha activation values. 
    
    Params:
        testInput dataclass, containing the dataframes, reference to the experiment configuration (may be None),
        and all necessary parameters. See the TestInput type annotation for more details.
    
    Returns:
        DataFrame: DataFrame containing all (metric, kf_reducer) combinations and the corresponding output data.
    
    """
    
    return post_experiment_test_grad(replace(ti, expected_ndims = 2))