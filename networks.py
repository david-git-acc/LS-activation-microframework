from abc import ABC, abstractmethod
from torch import nn
import torch
from typing import Callable
import numpy as np
from math import ceil, floor

class ActivationNetwork(ABC, nn.Module) :
    
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) : 
        super().__init__()    
    
        self.n_inputs : int = n_inputs
        self.n_outputs : int = n_outputs
        self.activation : type[nn.Module] = activation
        self._structure : nn.Sequential | None = None

    @property
    @abstractmethod    
    def structure(self) -> nn.Sequential :
        error_msg = "No sequential structure defined for ActivationNetwork inheriting subclass"
        assert isinstance(self._structure, nn.Sequential), error_msg 
        
        return self._structure
    
    def clear_activation_data(self) -> None :
        self.activation_grads : dict[int, torch.Tensor] = {}
        self.activation_outs : dict[int, torch.Tensor] = {}
    
    def create_activation_hook(self, layer_index : int = 0) -> Callable :
        
        def layer_activation_hook(module, inp, out) -> None :
            if self.training :
                
                if isinstance(out, tuple) :
                    result = out[0].detach().cpu()   
                else : 
                    result =  out.detach().cpu()     
                self.activation_outs[layer_index] = result     
        
        return layer_activation_hook
    
    def create_activation_grad_hook(self, layer_index : int = 0) -> Callable :
        
        def layer_activation_grad_hook(module, grad_in, grad_out) -> None :
            # Denied from using type hints due to unreasonable type structure - torch hooks really are a mess
            if self.training :
                
                if isinstance(grad_out, tuple) :
                    self.activation_grads[layer_index] = grad_out[0].detach().cpu()
                else :
                    self.activation_grads[layer_index] = grad_out.detach().cpu()
                
        return layer_activation_grad_hook
    
    def get_activations(self) -> dict[int, nn.Module] :
        activations = {}
        for layer_index, layer in enumerate( layer for layer in self.structure 
                                            if isinstance(layer, self.activation) ) :
            activations[layer_index] = layer
            
        return activations
        

    def create_all_activation_hooks(self) -> None :
        self.clear_activation_data()
        for layer_index, layer in enumerate( layer for layer in self.structure 
                                            if isinstance(layer, self.activation) ) :
            layer.register_full_backward_hook(self.create_activation_grad_hook(layer_index))
            layer.register_forward_hook(self.create_activation_hook(layer_index))
    
    def layer_widths(self) -> list[int] :
        
        widths = []
        for layer in self.structure :
            if hasattr(layer, "out_features") :
                widths.append(layer.out_features)
        
        return widths
    
    @property
    def width(self) -> int :
        
        return max(self.layer_widths())
    
    @property
    def n_activations(self) -> int :
        
        return len( self.get_activations() )
    
    @property
    def length(self) -> int :
        
        return len(self.layer_widths())
        
    @abstractmethod
    def forward(self, X) :
        if self.training :
            self.clear_activation_data()
            
        evaluated = self.structure(X)
        return evaluated


class ShortNetwork(ActivationNetwork) :
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self._structure = nn.Sequential(
            nn.Linear(n_inputs, 5),
            activation(),
            nn.Linear(5, 10),
            activation(),
            nn.Linear(10, 6),
            activation(),
            nn.Linear(6, n_outputs)
        )
    
        self.create_all_activation_hooks() 

    @property
    def structure(self) -> nn.Sequential : 
        return super().structure

    def forward(self, X) :
        return super().forward(X)
    
    
class DiamondNetwork(ActivationNetwork) :
    
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1, 
                 full_length : int = 20, max_width : int = 50) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self.full_length = full_length
        self.max_width = max_width
        self._structure = self.generate_structure()
        
        self.create_all_activation_hooks() 
        
    def generate_structure(self) -> nn.Sequential :
        # We add full_length + 1 layers so there are exactly full_length layers
        left_widths = np.geomspace(self.n_inputs, self.max_width, floor(self.full_length / 2) + 1, endpoint = True)
        right_widths = np.geomspace(self.max_width, self.n_outputs, ceil(self.full_length / 2), endpoint = True)
        layer_lengths = left_widths.astype(int).tolist() + right_widths.astype(int).tolist() 
        
        structure = []
        for layer_index, layer_length in enumerate(layer_lengths[1:], start = 1) :
            
            prior_layer_length = layer_lengths[layer_index - 1]
            layer_itself = nn.Linear(prior_layer_length, layer_length)
            
            # Avoid += logic to avoid O(l^2) concat operations
            structure.append(layer_itself)
            structure.append(self.activation())
            
        # We added an extra activation at the end so get rid of it
        structure.pop()
        
        return nn.Sequential(*structure)
    
    @property
    def structure(self) -> nn.Sequential : 
        return super().structure

    def forward(self, X) :
        return super().forward(X)