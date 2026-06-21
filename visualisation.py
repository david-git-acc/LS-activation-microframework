import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
from helpers import *

def get_n_colours(n : int, cmap : str = "viridis" ) -> np.ndarray :
    
    cmap_function = plt.get_cmap(cmap)
    
    c_range = np.linspace(0, 1, n)
    
    return cmap_function(c_range)


def plot_activation_data(activation_df : pd.DataFrame, figsize_px : tuple[float, float], 
                         plots_per_row : int = 3, linestyles : list[str] = ["-", "--", "-.", ":"],
                         exclude : set[str] = set(["activation", "index"]), save : bool = True, 
                         show : bool = False, savename : str = "plot.png",
                         max_samples : int = 100, markersize : float = 1,
                         n_skip : int = 0, title : str | None = None, kde : bool = True) -> None :

    activation_types = pd.unique(activation_df["activation"]).tolist()
        
    all_tests = [test for test in activation_df.columns if test not in exclude]
    
    # Used a custom bracket-encoding scheme from helpers to avoid multi-indexing in Pandas
    test_types, agg_types = extract_bencoded_list(all_tests, as_lists = True)
    
    n_tests = len(test_types)
    n_agg_types = len(agg_types)
    
    if len(linestyles) < n_agg_types :
        raise IndexError(f"Not enough linestyles ({len(linestyles)}) for number of aggfuncs ({n_agg_types})")
    
    nrows = ceil(n_tests / plots_per_row) 
    ncols = min(plots_per_row, n_tests)
    
    measure = str(activation_df.index.name)
    
    # Visualisation type changes depending on if we measured on epochs, parameters or neither
    uses_epochs = measure == "epoch"
    
    # Shown in the plot axis labels
    official_measurement_strname = measure.capitalize() + "s"
    
    fig, axes = plt.subplots(figsize = tuple(x / 96 for x in figsize_px), nrows = nrows, ncols = ncols)
    
    # Avoid 2D array for ease of implementation
    axes = np.atleast_1d(axes).flatten()
        
    # Map each test to a specific axis object so we can reference the same axis (n_agg_types) times
    test2ax = dict(zip(test_types, axes))
    
    # Each type of aggregation type has its own style for visual clarity
    agg2linestyle = dict(zip(agg_types, linestyles))
    
    activation_colours = get_n_colours(len(activation_types))

    for test in all_tests :
        
        test_type, agg_type = bdecode_name_pair(test)
        
        ax = test2ax[test_type]
        
        for j, activation_name in enumerate( activation_types ) :
            
            for_this_activation = activation_df["activation"] == activation_name    
            full_activation_data = activation_df[for_this_activation][n_skip : ] # First few are unstable
            
            sample_rate = ceil(len(full_activation_data) / max_samples)
            
            # Cut the number of datapoints for clarity
            activation_data = full_activation_data.iloc[::sample_rate]

            x = activation_data[test].index.tolist()
            y = activation_data[test]
                
            activation_colour = activation_colours[j]
            linestyle = agg2linestyle[agg_type]
            label = f"fold-{agg_type}({activation_name})"

            # Symmetric log scale - needs to be sign-preserving for negative values + 0
            y = symlog(y)

            if uses_epochs :
                
                ax.plot(x,y, label = label, 
                        color = activation_colour,
                        linestyle = linestyle,
                        marker = "^", markersize = markersize)
                
                # ax.fill_between(x, y_min, y, color = activation_colour, alpha = 0.3)
                
                ax.set_xlabel(official_measurement_strname, fontsize = "large")
                ax.set_ylabel(test_type, fontsize = "large")
                
            else :
                if kde: sns.kdeplot(x=y, color = activation_colour, linestyle = linestyle,
                            label = label, ax = ax,
                            fill = True, alpha = 0.3, common_norm = False)
                else: sns.histplot(x=y, color = activation_colour, linestyle = linestyle,
                            label = label, ax = ax,
                            fill = True, alpha = 0.3, common_norm = False,
                            stat = "probability")
                
                ax.set_xlabel(test_type, fontsize = "large")
                ax.set_ylabel("frequency density", fontsize = "large")
                
    for ax in axes :
        ax.grid(True, zorder = -1)
        ax.axhline(0, 0, 1, linestyle = "--", color = "red")
        ax.legend(loc = "upper left", prop = {'size': 9})          
        
    if title is None : title = f"Activation tests over {measure} on {", ".join(activation_types)}"
    
    plt.suptitle(title, fontsize = "xx-large")
    plt.tight_layout()
    
    if save : plt.savefig(savename)
    if show : plt.show()


   
