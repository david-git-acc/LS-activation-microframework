import torch



# REQUIREMENTS FOR REDUCER FUNCTIONS
# Reducers should have two key properties;
# P1. Should be generalisable to any arbitrary dimension or subset (dims can be int or tuple)
# P2. Should be able to handle NaN values gracefully without propagation

def variance(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    if isinstance(dim, int) :
        dim = (dim, )
    
    mean = torch.nanmean(X, dim=dim, keepdim = True)

    squares = (X - mean)**2

    return torch.nanmean(squares, dim = dim)

def stdeviation(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    return torch.sqrt(variance(X, dim))

def arithmetic_mean(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    meaned = torch.nanmean(X, dim = dim)

    return meaned    

def log_average(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    # Avoid negative badness
    safe = torch.abs(X) + 1e-10
    
    logged = torch.log(safe)
    
    return torch.nanmean(logged, dim = dim)

def donothing_dummy(x : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    return x

def last_elem(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    if isinstance(dim, tuple) :
        # If a tuple is passed, just 
        dims_left = [i for i in range(X.ndim) if i not in set(dim)]
        
        match len(dims_left) :
            case 0 :
                raise ValueError("Attempted to reduce over all possible dimensions simultaneously")
            case 1 : 
                dim = dims_left[0]
            case _ :
                dim = dims_left[0]
                print(f"Warning: attempted to select over ({len(dims_left)}) dimensions. Defaulting to dim {dim}.")

    return X.select(dim = dim, index = -1)

def norm(X : torch.Tensor, dim : int | tuple = 0) -> torch.Tensor :
    
    return torch.linalg.norm(X, dim = dim)