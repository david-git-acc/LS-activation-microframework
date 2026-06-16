### FUNCTIONS ###

from torch import nn

class ShortNetwork(nn.Module) :
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.structure = nn.Sequential(
            nn.Linear(n_inputs, 5),
            activation,
            nn.Linear(5, 10),
            activation,
            nn.Linear(10, 5),
            activation,
            nn.Linear(5, n_outputs)
        )
        
    def forward(self, X) :
        
        evaluated = self.structure(X)
        return evaluated