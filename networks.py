from abc import ABC, abstractmethod
from torch import nn
from typing import Callable
import torch

class ActivationNetwork(ABC, nn.Module) :
    
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) : 
        super().__init__()    
    
        self.n_inputs : int = n_inputs
        self.n_outputs : int = n_outputs
        self.activation : nn.Module = activation
        self.activation_outs : list[torch.Tensor] = []
        self.activation_grads : list[torch.Tensor] = []

    @property
    @abstractmethod    
    def structure(self) -> nn.Sequential :
        pass
    
    def clear_activation_data(self) -> None :
        self.activation_grads = []
        self.activation_outs = []
    
    def create_activation_hook(self, module, inp, out) -> None :
        self.activation_outs.append(out[0].detach().cpu())      
    
    def create_activation_grad_hook(self, module, grad_in, grad_out) -> None :
        # Denied from using type hints due to unreasonable type structure - torch hooks really are a mess
        self.activation_grads.append(grad_out[0].detach().cpu())

    def create_all_activation_hooks(self) -> None :
        for layer in self.structure :
            if isinstance(layer, type(self.activation)) :
                layer.register_full_backward_hook(self.create_activation_grad_hook)
                layer.register_forward_hook(self.create_activation_hook)
    
    def layer_widths(self) -> list[int] :
        
        widths = []
        for layer in self.structure :
            if hasattr(layer, "out_features") :
                widths.append(layer.out_features)
        
        return widths
    
    def width(self) -> int :
        
        return max(self.layer_widths())
    
    def length(self) -> int :
        
        return len(self.layer_widths())
        
    
    @abstractmethod
    def forward(self, X) :
        self.clear_activation_data()
        evaluated = self.structure(X)
        return evaluated


class ShortNetwork(ActivationNetwork) :
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self._structure = nn.Sequential(
            nn.Linear(n_inputs, 5),
            activation,
            nn.Linear(5, 10),
            activation,
            nn.Linear(10, 5),
            activation,
            nn.Linear(5, n_outputs),
            activation
        )
    
        self.create_all_activation_hooks() 
    
    @property
    def structure(self) -> nn.Sequential:
        return self._structure

    def forward(self, X) :
        return super().forward(X)