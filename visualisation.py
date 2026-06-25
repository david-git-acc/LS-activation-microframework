import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
from helpers import *
from config import kfold_k
from dataclass_objects import plotParams

def get_n_colours(n : int, cmap : str = "viridis" ) -> np.ndarray :
    
    cmap_function = plt.get_cmap(cmap)
    
    c_range = np.linspace(0, 1, n)
    
    return cmap_function(c_range)


def plot_activation_data(activation_df : pd.DataFrame, figsize_px : tuple[float, float], 
                         plots_per_row : int = 3, linestyles : list[str] = ["-", "--", "-.", ":"],
                         exclude : set[str] = set(["activation", "index"]), save : bool = True, 
                         show : bool = False, savename : str = "plot.png",
                         max_samples : int = 100, markersize : float = 1,
                         n_skip : int = 0, title : str = "", plot_type : str = "epochs") -> None :

    activation_types = pd.unique(activation_df["activation"]).tolist()
        
    all_tests = [test for test in activation_df.columns if test not in exclude]
    test_types, agg_types = extract_tuple_list([col for col in activation_df.columns if isinstance(col, tuple)])
     
    nrows = ceil(len(test_types) / plots_per_row) 
    ncols = min(plots_per_row, len(test_types))
    
    fig, axes = plt.subplots(figsize = tuple(x / 96 for x in figsize_px), nrows = nrows, ncols = ncols)
    
    # Avoid 2D array for ease of implementation
    axes = np.atleast_1d(axes).flatten()

    # Map each test to a specific axis object so we can reference the same axis (n_agg_types) times
    test2ax = dict(zip(test_types, axes))
    
    # Each type of aggregation type has its own style for visual clarity
    agg2linestyle = dict(zip(agg_types, linestyles))
    
    activation_colours = get_n_colours(len(activation_types))

    for test_type, agg_type in all_tests :
        ax = test2ax[test_type]
        
        for j, activation_name in enumerate( activation_types ) :
            
            for_this_activation = activation_df["activation"] == activation_name    
            full_activation_data = activation_df[for_this_activation][n_skip : ] # First few are unstable
            
            sample_rate = ceil(len(full_activation_data) / max_samples)
            
            # Cut the number of datapoints for clarity
            activation_data = full_activation_data.iloc[::sample_rate]

            x = activation_data[(test_type, agg_type)].index.tolist()
            y = activation_data[(test_type, agg_type)]
            
            params = plotParams(activation_name, plot_type, activation_colours[j], agg2linestyle.get(agg_type, "-"), 
                                xlabel = str(activation_df.index.name), ylabel = test_type, 
                                legend_label = f"fold-{agg_type}({activation_name})")
                
            plot_activation(x, y, ax, params)
                
    for ax in axes :
        ymin, ymax = ax.get_ylim()
        if 0 > ymin and 0 < ymax : 
            ax.axhline(0, 0, 1, linestyle = "--", color = "red")
         
    plt.suptitle(title, fontsize = "xx-large")
    plt.tight_layout()
    
    if save : plt.savefig(savename)
    if show : plt.show()
    
    plt.close(fig)
    
    
    
def plot_activation(x, y, ax, p : plotParams = plotParams()) :
        
    if p.plot_type == "curve" :
        
        # Symmetric log scale - needs to be sign-preserving for negative values + 0
        y = symlog(y)
        
        ax.plot(x, y, label = p.legend_label, color = p.colour, linestyle = p.linestyle, marker = p.marker, 
                markersize = p.markersize )
        ax.set_xlabel(p.xlabel, fontsize = "large")
        ax.set_ylabel(p.ylabel, fontsize = "large")
        
    elif p.plot_type == "kde" : 
        sns.kdeplot(x=y, color = p.colour, linestyle = p.linestyle, label = p.legend_label, ax = ax, alpha = 0.3,
                    fill = True, common_norm = False)
    else: 
        sns.histplot(x=y, color = p.colour, linestyle = p.linestyle, label = p.legend_label, ax = ax,
                    fill = True, alpha = 0.3, common_norm = False,  stat = "probability")
        
    if p.plot_type != "curve" :   
        ax.set_xlabel(p.ylabel, fontsize = "large")
        ax.set_ylabel("frequency density", fontsize = "large")
        
    ax.grid(True, zorder = -1)
    ax.legend(loc = "upper left", prop = {'size': 9})    
    
    
        
def determine_plot_type(eval_type, category, measure_type) -> str :
    
    if measure_type == "epochs" :
        return "curve"
    elif eval_type == "train" : 
        return "histplot"
    else :
        return "kde"
    
def plot_activation_tests(results : dict[tuple[str, str, str], pd.DataFrame] , 
                          save_csv_folder : str = "saved_csvs", save_fig_folder : str = "saved_figures") -> None :

    save_path_csv = f"{generate_savefolder(results)}/{save_csv_folder}"
    save_path_fig = f"{generate_savefolder(results)}/{save_fig_folder}"
    
    create_path(save_path_csv)
    create_path(save_path_fig)

    for eval_type, category, measure_type in results :

        total_activation_df = results[(eval_type, category, measure_type)]
        savename = f"{category}-{measure_type}_on_{eval_type}"
            
        total_activation_df.to_csv(f"{save_path_csv}/{savename}.csv")
            
        plot_activation_data(total_activation_df, figsize_px = (1920, 1080),
                        max_samples = 50, markersize = 4,
                        savename = f"{save_path_fig}/{savename}.png",
                        title = generate_plot_title(category, 1 if eval_type == "test" else kfold_k),
                        n_skip = 5, plot_type = determine_plot_type(eval_type, category, measure_type))