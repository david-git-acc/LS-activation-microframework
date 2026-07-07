import torch

def variance(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    if isinstance(dim, int) :
        dim = (dim, )
    
    mean = torch.nanmean(X, dim=dim, keepdim = True)

    squares = (X - mean)**2

    return torch.nanmean(squares, dim = dim)

def arithmetic_mean(X : torch.Tensor, dim : int = 0) -> torch.Tensor :
    
    meaned = torch.nanmean(X, dim = dim)

    return meaned    

def log_average(X : torch.Tensor, dim : int = 0) -> torch.Tensor :
    
    # Avoid negative badness
    safe = torch.abs(X) + 1e-10
    
    logged = torch.log(safe)
    
    return torch.nanmean(logged, dim = dim)

def testloss_dummy(x : torch.Tensor, dim : int = 0) -> torch.Tensor :
    
    return x