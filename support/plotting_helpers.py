import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def get_n_colours(n : int, cmap : str = "viridis" ) -> list :
    
    cmap_function = plt.get_cmap(cmap)
    
    c_range = np.linspace(0, 1, n)
    
    return cmap_function(c_range).tolist()

def determine_plot_type(eval_type, category, measure_type) -> str :
    
    if measure_type == "epochs" :
        return "curve"
    elif eval_type == "train" : 
        return "histplot"
    else :
        return "kde"
    
def generate_plot_title(category : str, kfold_k : int = 0) -> str :
    
    eval_type = "train" if kfold_k > 1 else "test"
    
    fold_explanation = f", {kfold_k}-fold" if kfold_k else ""
    
    title = f"[{eval_type} data{fold_explanation}] Activation tests over {category} data"
    
    return title      

def is_empty_axis(ax) -> bool :
    
    return not (ax.lines or ax.collections or ax.patches)

def df2csv(df : pd.DataFrame, filename : str) -> None :
    
    original_view = df.columns.copy()
    
    # Save as multiindex so pd won't register tuple column names as simple strings, which would destroy information
    df.columns = pd.MultiIndex.from_tuples([col if isinstance(col, tuple) else (col, ) for col in df.columns])
    df.to_csv(filename)
    
    # Prevent side effects
    df.columns = original_view