from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import torch
from torch import nn
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import copy

# CUSTOM
from networks import ActivationNetwork
from support.parsing_helpers import validate_activation_df_column_names
from dataclass_objects.config_objects import expConfig

@dataclass
class expInput() :
    
    """Input dataclass to be passed into experiment() or experiment_from_df() function. Used for validation, re-use and 
    to avoid congested function signatures,
    
    Params:
        X_train_tensor: n x d training feature matrix.
        X_test_tensor: n_test x d testing feature matrix.
        Y_train_tensor: 1 x n or n x 1 training label matrix.
        Y_test_tensor: 1 x n_test or n_test x 1 testing label matrix.
        anet_model: the model to perform the experiment with.
        target_loss: the loss function to evaluate the model.
        epochs: number of complete sweeps of X_train_tensor and Y_train_tensor to perform to train the model.
        lr: constant learning rate value. In theory, you could pass in a variable learning rate here (not recommended).
        batch_size: number of training examples to use per gradient descent step. Defaults to -1 (all examples per step).
        max_recorded_samples: maximum number of training steps to be recorded and captured in experimentResult. Defaults to -1 (all).
        categories: tuple of all category data that the experiment should track throughout the experiment. Defaults to grad.
    """
    
    X_train_tensor : torch.Tensor
    X_test_tensor : torch.Tensor
    Y_train_tensor : torch.Tensor
    Y_test_tensor : torch.Tensor 
    anet_model : ActivationNetwork
    target_loss : nn.Module = nn.CrossEntropyLoss()
    optim_type : type[torch.optim.Adam] | type[torch.optim.SGD] = torch.optim.Adam
    epochs : int = 500 
    lr : float = 0.001
    batch_size : int = -1
    max_recorded_samples : int = -1
    categories : tuple[str, ...] = ("grad",)
    device_thresh : int = 1
    
    def __post_init__(self) : 
        # Use full-batch GD if no batch size given
        self.batch_size = len(self.X_train_tensor) if self.batch_size == -1 else self.batch_size
        self.max_recorded_samples = self.epochs if self.max_recorded_samples == -1 else self.max_recorded_samples
        
        self.training_dataset = TensorDataset(self.X_train_tensor, self.Y_train_tensor)
        self.training_dataloader = DataLoader(self.training_dataset, self.batch_size, shuffle = True, pin_memory = True)

        self.optim = self.optim_type(self.anet_model.parameters(), lr = self.lr)
        self.saved_params : dict[str, Any] = {}
        
        # Below typical dataset sizes, GPU becomes slower than CPU
        self.device_type = self.preferred_device
        self.to_device(self.device_type)
        
        self.save_state()
        
    def save_state(self) -> None :
        self.saved_params["anet_model"] = { param : value.clone() 
                                           for param, value in self.anet_model.state_dict().items() }
        self.saved_params["optim"] = copy.deepcopy(self.optim.state_dict())
        self.saved_params["device_type"] = self.device_type
        self.saved_params["training"] = self.anet_model.training
    
    def reload_state(self, switch_device : bool = True) -> None :
        self.anet_model.load_state_dict(self.saved_params["anet_model"])
        self.optim.load_state_dict(self.saved_params["optim"])
        self.anet_model.train(self.saved_params["training"])

        if switch_device :
            self.device_type = self.saved_params["device_type"]
            self.to_device(self.device_type)

    def to_device(self, device_type : str, strict : bool = False) -> None :
        if device_type == "cuda" and not torch.cuda.is_available() :
            error_msg = "Tried to set to GPU mode, but CUDA is not available"
            if strict : raise ValueError(error_msg)
            else : print(f"Warning: {error_msg}")
        
        if self.anet_model._structure is None :
            raise ValueError(f"Network model structure for {type(self.anet_model)} is not defined")
        
        self.device = torch.device(device_type) 
        for torch_object in (self.anet_model, self.target_loss, self.anet_model._structure) :
            torch_object.to(self.device)
        
        # Necessary evil boilerplate
        self.X_train_tensor = self.X_train_tensor.to(self.device)
        self.X_test_tensor = self.X_test_tensor.to(self.device)
        self.Y_train_tensor = self.Y_train_tensor.to(self.device)
        self.Y_test_tensor = self.Y_test_tensor.to(self.device)
    
    @property 
    def preferred_device(self) -> str :
        if len(self.X_train_tensor) >= self.device_thresh and torch.cuda.is_available() :
            return "cuda"
        return "cpu"

    @property
    def n_captures(self) -> int :
        return min(self.epochs, self.max_recorded_samples)
    
    
@dataclass
class testInput() :
    """
    Dataclass to store inputs for post-experiment-test functions (e.g post_experiment_test_grad) for input validation
    and easy use.
    
    Params:
        X: Torch tensor data to be tested on. 
        reducers: list of functions to test on, e.g mean, variance, log_avg.
        reducer_names: names of each test. If no value given, uses the function names.
        kf_reducers: the aggregation functions to collapse a dimension over. Always becomes mean() if number of folds = 1.
        kf_reducer_names: names of aggregation functions.
        expected_ndims : number of dimensions that the data is originally meant to be in (before folds). Used for validation.
        measure_type: dimension to check over. Can be set to a different axis manually.
        metadata: dictionary containing any relevant data to be collected for future use without adding to the code.
        xpc: reference to the parent expConfig object. Can be None for most tasks, but for others must be set.
    
    """
    
    X : torch.Tensor
    reducers : tuple[Callable, ...]
    reducer_names : str | list[str]
    kf_reducers :  tuple[Callable, ...]
    kf_reducer_names : str | list[str]
    expected_ndims : int = 2
    measure_type : str = "epochs"
    metadata : dict[str, Any] = field(default_factory = dict, init = True)
    xpc : expConfig | None = None
    
    def __post_init__(self) :
      
        if not isinstance(self.kf_reducers, tuple) : self.kf_reducers = ( self.kf_reducers, )  
        if not isinstance(self.reducer_names, list) : self.reducer_names = [self.reducer_names]
        if not isinstance(self.kf_reducer_names, list) : self.kf_reducer_names = [self.kf_reducer_names]
        
        is_test_data = len(self.X.size()) == self.expected_ndims
        
        # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
        if is_test_data :
            self.X = self.X[..., None]
        
        self.reducer_names = validate_activation_df_column_names(self.reducers, self.reducer_names)
        self.kf_reducer_names = validate_activation_df_column_names(self.kf_reducers, self.kf_reducer_names)