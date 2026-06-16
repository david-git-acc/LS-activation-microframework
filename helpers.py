def list2dict(pair_data : list[tuple[list[str], object]]) -> dict :
    
    mapping = {}
    
    for columns, val in pair_data :
        
        for column in columns : 
            
            mapping[column] = val
    
    return mapping
    
def index_name(i : int = 0) -> str :
    
    mapping = {
        0 : "param",
        1 : "epoch",
    }
    
    return mapping.get(i, "iter")

import torch
from torch.utils.data import Dataset
import pandas as pd

class PandasDataset(Dataset) :
    
    def __init__(self, df : pd.DataFrame, feature_cols : list[str], target_cols : list[str], 
                 x_type : torch.dtype, y_type : torch.dtype) :
        
        if isinstance(target_cols, str) :
            target_cols = [target_cols]
            
        if isinstance(feature_cols, str) :
            feature_cols = [feature_cols]
        
        self.data = df[feature_cols].values
        self.target = df[target_cols].values
        
        # For conversion to pytorch compatible datatypes
        self.x_type = x_type
        self.y_type = y_type
        
    
    def __len__(self) -> int :
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = torch.tensor( self.data[idx], dtype = self.x_type)
        y = torch.tensor( self.target[idx], dtype = self.y_type)
        
        return (x,y)