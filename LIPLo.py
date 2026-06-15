import torch
from torch import nn

class LIPLo_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, torch.tensor(alpha))
        # log1p(x) is log(1 + x), which is more numerically stable
        return alpha * x + (1 - alpha) * torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        # Derivative calculation
        derivative = alpha + (1 - alpha) / (1 + torch.abs(x))
        return grad_output * derivative, None  # Return None for the 'alpha' gradient

class LIPLo(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return LIPLo_.apply(x, self.alpha)