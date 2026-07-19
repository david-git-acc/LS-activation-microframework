from abc import ABC, abstractmethod
from torch import nn
import torch
from typing import Callable
import numpy as np
from math import ceil, floor

class ActivationNetwork(ABC, nn.Module) :
    
    def __init__(self, activation : type[nn.Module], n_inputs : int = 1, n_outputs : int = 1) : 
        super().__init__()    
        
        self._name = ""
        self.n_inputs : int = n_inputs
        self.n_outputs : int = n_outputs
        self.activation : type[nn.Module] = activation
        self._structure : nn.Sequential | None = None
        self.recording = False
        self.hook_structures = []
        
        self.clear_activation_data()

    @abstractmethod
    def generate_structure(self) -> nn.Sequential :
        pass

    @property
    def name(self) -> str :
        if not self._name : 
            raise ValueError("Name not instantiated for activation network")
        
        return self._name

    @property   
    def structure(self) -> nn.Sequential :
        error_msg = "No sequential structure defined for ActivationNetwork inheriting subclass"
        assert isinstance(self._structure, nn.Sequential), error_msg 
        
        return self._structure
    
    def clear_activation_data(self) -> None :
        self.activation_grads : dict[int, torch.Tensor] = {}
        self.activation_outs : dict[int, torch.Tensor] = {}
    
    def clear_hooks(self) -> None :
    
        for hook_struct in self.hook_structures : 
            hook_struct.remove()
        self.hook_structures = []
        
    def record(self, mode : bool = True) -> None :
        self.recording = mode
    
    def create_activation_hook(self, layer_index : int = 0) -> Callable :
        
        def layer_activation_hook(module, inp, out) -> None :
            if self.training and self.recording :
                
                result : torch.Tensor = out[0] if isinstance(out, tuple) else out
                self.activation_outs[layer_index] = result.detach().cpu()
                
        return layer_activation_hook
    
    def create_activation_grad_hook(self, layer_index : int = 0) -> Callable :
        
        def layer_activation_grad_hook(module, grad_in, grad_out) -> None :
            # Denied from using type hints due to unreasonable type structure - torch hooks really are a mess
            if self.training and self.recording :
                
                result : torch.Tensor = grad_out[0] if isinstance(grad_out, tuple) else grad_out
                self.activation_grads[layer_index] = result.detach().cpu()
                      
        return layer_activation_grad_hook
    
    
    def get_activations(self) -> dict[int, nn.Module] :
        activations = {}
        for layer_index, layer in enumerate( layer for layer in self.structure 
                                            if isinstance(layer, self.activation) ) :
            activations[layer_index] = layer
            
        return activations
        
        
    def create_all_activation_hooks(self) -> None :
        self.clear_activation_data()
        self.clear_hooks() # Prevent accumulation of hooks
        for layer_index, layer in enumerate( layer for layer in self.structure 
                                            if isinstance(layer, self.activation) ) :
            grad_hook_struct = layer.register_full_backward_hook(self.create_activation_grad_hook(layer_index))
            out_hook_struct = layer.register_forward_hook(self.create_activation_hook(layer_index))
    
            # Track to remove in next call. I don't know why we have to do it like this, it's how torch works
            self.hook_structures.append(grad_hook_struct)
            self.hook_structures.append(out_hook_struct)
    
    def layer_widths(self) -> list[int] :
        
        widths = []
        for layer in self.structure :
            if hasattr(layer, "out_features") :
                widths.append(layer.out_features)
        
        return widths
  
    @property
    def n_activations(self) -> int :
        
        return len( self.get_activations() )
    
    @property
    def width(self) -> int :
        
        return max(self.layer_widths())
    
    @property
    def length(self) -> int :
        
        return len(self.layer_widths())
        
    def forward(self, X : torch.Tensor) -> torch.Tensor :
        if self.training :
            self.clear_activation_data()
            
        evaluated = self.structure(X)
        return evaluated


class ShortNetwork(ActivationNetwork) :
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self._name = "ShortNetwork"
        self._structure = self.generate_structure()
        self.create_all_activation_hooks() 

    def generate_structure(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.n_inputs, 5),
            self.activation(),
            nn.Linear(5, 10),
            self.activation(),
            nn.Linear(10, 6),
            self.activation(),
            nn.Linear(6, self.n_outputs)
        )

class DiamondNetwork(ActivationNetwork) :
    
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1, 
                 full_length : int = 15, max_width : int = 100) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self._name = "DiamondNetwork"
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

def to_batchnorm(anet : type[ActivationNetwork]) -> type[ActivationNetwork] :
    
    class BatchNormNetwork(ActivationNetwork) :
        
        # Unlike in to_LS, we keep all the non-batchnorm-related specifics algebraic
        # so we can compose with other transforms e.g to_layernorm (future)
        def __init__(self, activation : type[nn.Module], n_inputs : int = 1, n_outputs : int = 1) :
            super().__init__(activation, n_inputs, n_outputs)
            
            # Instantiate a dummy so we can access the class-specific shared attributes
            self.base = anet 
            self._name = "BN-" + anet(nn.Identity, 1, 1).name
            self._structure = self.generate_structure()
            self.create_all_activation_hooks() 
            
        def generate_structure(self) -> nn.Sequential :

            current_structure = self.base(self.activation, self.n_inputs, self.n_outputs).structure
            new_structure = []
            
            for layer_index in range(len(current_structure) - 1) :
                
                current_layer = current_structure[layer_index]
                next_layer = current_structure[layer_index + 1]
                
                new_structure.append(current_layer)
                
                if isinstance(current_layer, nn.Linear) and isinstance(next_layer, self.activation) :
                    batchnorm = nn.BatchNorm1d(current_layer.out_features, affine = True)
                    new_structure.append(batchnorm)

            new_structure.append(current_structure[-1])
            
            return nn.Sequential(*new_structure)
    
    return BatchNormNetwork