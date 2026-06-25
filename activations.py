import torch
from torch import nn
from helpers import get_name

# class LIPLo_(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.save_for_backward(x, torch.tensor(alpha))
#         # log1p(x) is log(1 + x), which is more numerically stable
#         return alpha * x + (1 - alpha) * torch.sign(x) * torch.log1p(torch.abs(x))

#     @staticmethod
#     def backward(ctx, grad_output): # type: ignore
#         x, alpha = ctx.saved_tensors
#         # Derivative calculation
#         derivative = alpha + (1 - alpha) / (1 + torch.abs(x))
#         return grad_output * derivative, None  # Return None for the 'alpha' gradient

class IPLo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return LIPLo_.apply(x, self.alpha)
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
class LS(nn.Module) :
    
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