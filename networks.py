from abc import ABC, abstractmethod
from torch import nn


class ActivationNetwork(ABC, nn.Module) :
    
    def __init__(self, activation, structure, n_inputs : int = 1, n_outputs : int = 1) : 
        super().__init__()    
    
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.structure = structure
    
    @abstractmethod
    def forward(self, X) :
        
        evaluated = self.structure(X)
        return evaluated


class ShortNetwork(ActivationNetwork) :
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
        structure = nn.Sequential(
            nn.Linear(n_inputs, 5),
            activation,
            nn.Linear(5, 10),
            activation,
            nn.Linear(10, 5),
            activation,
            nn.Linear(5, n_outputs)
        )
        
        super().__init__(activation, structure, n_inputs, n_outputs)

    def forward(self, X) :
        return super().forward(X)