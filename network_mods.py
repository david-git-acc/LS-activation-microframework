from abc import ABC, abstractmethod
from torch import nn
import torch
from typing import Callable
import numpy as np
from math import ceil, floor

### CUSTOM
from networks import ActivationNetwork

def update_structure(anet : type[ActivationNetwork], 
                     update_func : Callable, update_name : str) -> type[ActivationNetwork] :
    
    class InsertionNetwork(ActivationNetwork) :
        
        _name = update_name + "-" + anet._name
        
        def __init__(self, activation : type[nn.Module], n_inputs : int = 1, n_outputs : int = 1) :
            super().__init__(activation, n_inputs, n_outputs)
            
            # Instantiate a dummy so we can access the class-specific shared attributes
            self.base = anet 
            self._structure = self.generate_structure()
            self.create_all_activation_hooks() 
            
        def generate_structure(self) -> nn.Sequential :

            current_structure = self.base(self.activation, self.n_inputs, self.n_outputs).structure
            new_structure = insert_layers(current_structure, self, update_func)

            return new_structure
    
    return InsertionNetwork


def insert_layers(current_structure : nn.Sequential, network_instance : ActivationNetwork,
                    update_func : Callable) -> nn.Sequential :
    
        new_structure = []      
        for layer_index in range(len(current_structure) - 1) :
            
            current_layer = current_structure[layer_index]   
            new_structure.append(current_layer)
            
            new_component = update_func(network_instance, current_structure, layer_index) 
            
            if new_component is None : 
                continue
            elif not isinstance(new_component, nn.Module) : 
                error_msg = f"Structural insertion must return nn.Module or None, not {type(new_component)}"
                raise ValueError(error_msg)
            
            new_structure.append(new_component)

        new_structure.append(current_structure[-1])
        
        return nn.Sequential(*new_structure)


def to_batchnorm(anet : type[ActivationNetwork]) -> type[ActivationNetwork] :
    
    def update_func(network : ActivationNetwork, current_structure : nn.Sequential, 
                    layer_index : int) -> nn.Module | None :
        
        current_layer = current_structure[layer_index]
        if isinstance(current_layer, nn.Linear) :
            return nn.BatchNorm1d(current_layer.out_features)

        return None            
    
    return update_structure(anet, update_func, "BN")


def to_layernorm(anet : type[ActivationNetwork]) -> type[ActivationNetwork] :
       
    def update_func(network : ActivationNetwork, current_structure : nn.Sequential, 
                    layer_index : int) -> nn.Module | None :
        
        current_layer = current_structure[layer_index]
        if isinstance(current_layer, nn.Linear) :
            return nn.LayerNorm(current_layer.out_features)

        return None  
         
    return update_structure(anet, update_func, "LN")

def to_dropout(anet : type[ActivationNetwork], p : float = 0.5) -> type[ActivationNetwork] :
    
    def update_func(network : ActivationNetwork, current_structure : nn.Sequential, 
                    layer_index : int) -> nn.Module | None :
        
        current_layer = current_structure[layer_index]
        if isinstance(current_layer, network.activation) :
            return nn.Dropout(p = p, inplace = False)
        
        return None

    return update_structure(anet, update_func, f"Drop{int(p*100)}")

class ResidualConnection(nn.Module) :
    
    def __init__(self, structure : nn.Sequential ) :
        super().__init__()

        self.structure = structure
        linear_layers = [layer for layer in structure if isinstance(layer, nn.Linear)]
        
        self.proj = nn.Identity()
        if linear_layers :
            self.in_features = linear_layers[0].in_features
            self.out_features = linear_layers[-1].out_features
            
            if self.in_features != self.out_features : 
                self.proj = nn.Linear(self.in_features, self.out_features) 

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        return self.structure(X) + self.proj(X)
    
def to_residual(anet : type[ActivationNetwork], block_size : int = 3) -> type[ActivationNetwork] :
      
    
    # Would not fit with the existing linear-like helper functions, so had to make from scratch
    class ResidualNetwork(ActivationNetwork) :
        
        _name = f"Skip{block_size}-" + anet._name
        
        def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
            super().__init__(activation, n_inputs, n_outputs)        
            self.base = anet
            self.block_size = block_size
            self._structure = self.generate_structure()
            self.create_all_activation_hooks() 

        def generate_structure(self) -> nn.Sequential:
            
            current_structure = self.base(self.activation, self.n_inputs, self.n_outputs).structure
            new_structure = []
            up_to_activation = []
            current_block = []
            units = []
           
            for layer in current_structure :
                up_to_activation.append(layer)
                
                if isinstance(layer, self.activation) :
                    units.append(up_to_activation)
                    up_to_activation = []
            
            # Add any remainder    
            units.append(up_to_activation)
            
            for unit in units :
                current_block.append(unit)
                
                if len(current_block) >= block_size :
                    flat_structure = [layer for unit in current_block for layer in unit]
                    sequence = nn.Sequential(*flat_structure)
                    new_structure.append(ResidualConnection(sequence))
                    current_block = []
            
            new_structure += [layer for unit in current_block for layer in unit]
            
            return nn.Sequential(*new_structure)
    
    return ResidualNetwork