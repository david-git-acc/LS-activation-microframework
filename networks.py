from abc import ABC, abstractmethod
from torch import nn
import torch
from typing import Callable

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
        self.activation_grads : list[torch.Tensor] = [torch.Tensor([]) for _ in range(self.length)]
        self.activation_outs : list[torch.Tensor] = [torch.Tensor([]) for _ in range(self.length)]
    
    def create_activation_hook(self, layer_index : int = 0) -> Callable :
        
        def layer_activation_hook(module, inp, out) -> None :
            if self.training :
                self.activation_outs[layer_index] = out[0].detach().cpu()      
        
        return layer_activation_hook
    
    def create_activation_grad_hook(self, layer_index : int = 0) -> Callable :
        
        def layer_activation_grad_hook(module, grad_in, grad_out) -> None :
            # Denied from using type hints due to unreasonable type structure - torch hooks really are a mess
            if self.training :
                self.activation_grads[layer_index] = grad_out[0].detach().cpu()
                
        return layer_activation_grad_hook

    def create_all_activation_hooks(self) -> None :
        self.clear_activation_data()
        for layer_index, layer in enumerate( layer for layer in self.structure if isinstance(layer, self.activation) ) :
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
            nn.Linear(6, n_outputs),
            activation()
        )
    
        self.create_all_activation_hooks() 

    @property
    def structure(self) -> nn.Sequential : 
        return super().structure

    def forward(self, X) :
        return super().forward(X)