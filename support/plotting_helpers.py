import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def get_n_colours(n : int, cmap : str = "viridis" ) -> list :
    
    cmap_function = plt.get_cmap(cmap)
    
    c_range = np.linspace(0, 1, n)
    
    return cmap_function(c_range).tolist()

def determine_plot_type(eval_type, category, measure_type) -> str :
    
    if measure_type == "epochs" or measure_type == "layers":
        return "curve"
    elif eval_type == "train" : 
        return "histplot"
    else :
        return "kde"

def category2name(category : str) -> str :
    
    mapping = {
        "grad" : "gradient",
        "agrad" : "activation gradient", 
        "testloss" : "test loss",
        "testpreds" : "test prediction",
        "aouts" : "activation output",
        "metrics" : "evaluation metric"
    }
    
    return mapping.get(category, category).title()
   
def is_empty_axis(ax) -> bool :
    
    return not (ax.lines or ax.collections or ax.patches)

def df2csv(df : pd.DataFrame, filename : str) -> None :
    
    original_view = df.columns.copy()
    
    # Save as multiindex so pd won't register tuple column names as simple strings, which would destroy information
    df.columns = pd.MultiIndex.from_tuples([col if isinstance(col, tuple) else (col, ) for col in df.columns])
    df.to_csv(filename)
    
    # Prevent side effects
    df.columns = original_view