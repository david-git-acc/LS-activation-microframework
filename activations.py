import torch
from torch import nn
from helpers import get_name

class IPLo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return LIPLo_.apply(x, self.alpha)
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
class LS(nn.Module) :
    
    """
        Superfunction that takes in a base activation function and applies the LS application to it:

            LS_{alpha}(f(x)) = alpha x + (1-alpha) (f(x) / f'(0))
    
        Apply LS to an existing activation and returns a new activation with this property and an initial alpha.
        If learnable = true, alpha can update with further training of the neural network.
        If learnable = false, untouchable (but exists inside parameters()). Useful for sensitivity analysis.
        
        Key part of the developed theory.
        
        Params:
            base_activation: original activation to apply LS transformation to.
            alpha: alpha value to choose in the LS transformation.
            learnable: whether alpha can be modified during training.
            dtype: datatype to use in computations of the activation.
            
    """
    
    def __init__(self, base_activation : nn.Module, alpha : float = 0.01, learnable : bool = False,
                 dtype : torch.dtype = torch.float32) :
        super().__init__()
        
        self.base_activation = base_activation
        self.original_alpha = alpha
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype = dtype, requires_grad = True), requires_grad = learnable)
        self.learnable = learnable
        self.dtype = dtype
        
        self.register_buffer("f_prime_0", self.calculate_f_prime_at_0(), persistent = True)
        
    def calculate_f_prime_at_0(self, epsilon_thresh : float = 1e-10) -> torch.Tensor :
        
        x = torch.tensor([1e-5], requires_grad = True)
        f_x = self.base_activation(x)
                
        grad_at_0 = torch.autograd.grad(f_x, x)[0].item()

        if abs(grad_at_0) <= epsilon_thresh :
            raise ValueError(f"Derivative of {get_name(self.base_activation)} at 0 is 0; not a valid function for LS conversion")

        return torch.tensor( grad_at_0, dtype = self.dtype)

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        
        a = torch.clip(self.alpha, min = 1e-10, max = 1 - 1e-10)
        
        out = a * X + (1-a) * ( self.base_activation(X) / self.f_prime_0 )
        
        return out