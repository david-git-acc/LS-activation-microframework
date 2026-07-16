import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from rich.progress import Progress
from typing import Any

### CUSTOM
from categories.base_definitions import is_ordinal
from dataclass_objects.config_objects import expVisual, expConfig
from dataclass_objects.result_objects import activationResults
from support.plotting_helpers import is_empty_axis, df2csv
from support.processing_helpers import symlog, sampling_indices
from support.parsing_helpers import create_path

matplotlib.use('Agg') # No interactive window, purely file-based rendering

def populate_axes(ax, ax_params : dict[str, Any]) -> None :

    """
    Represents work to be done to generate the axes before any plots/curves are drawn.
    """

    ax.set_xlabel(ax_params["xlabel"])
    ax.set_ylabel(ax_params["ylabel"])
    ax.grid(ax_params["grid"])
    
    # We only set x-ticks if there is an x-axis. For KDEs/histplots, y-axis becomes x-axis so no need
    if ax_params["xaxis"] :
        # +1 because normally this function captures indices and we go from 1-n, not 0-n-1
        xticks = sampling_indices(max(ax_params["xaxis"]) + 1, ax_params["nxticks"])
        ax.set_xticks(ticks = xticks)
    
def post_plotting_axes_ops(ax) -> None :
    
    """
    Represents work to be done after all axes plots/curves have been drawn.
    """
    
    if is_empty_axis(ax) :
        ax.figure.delaxes(ax)
        return
    
    ax.legend(loc = "upper left", fontsize = 9)   
     
    # Show the x-axis for reference
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax : 
        ax.axhline(0, 0, 1, linestyle = "--", color = "red")
    
def plot_data(x : np.ndarray, y, ax, plot_params : dict[str, Any], apply_symlog : bool = True ) -> None :
    # Can't add type to y or linter cries irrationally
    
    """
    Given x and y data and parameters, create a plot on a given axes object.
    Curves are symlogged via the function

        symlog(x) = sgn(x)ln(1+|x|)
    
    which is linear around the origin but compresses data magnitudes while preserving sign for visual clarity.
    Training data over parameters/non-enumerable data is drawn as histplot rather than KDE for visual stability.
    
    Params:
        x: x-axis data.
        y: y-axis data. Must equal length of x.
        ax: matplotlib axes object reference to draw the plot on.
        plot_params: information about the plot for drawing it, e.g legends, colours, linestyles.
    
    """
    # Switch-case offers only marginal speedup gains over branching, but more readable + extensible
    match plot_params["plot_type"] :
        case "curve" :
            # Symmetric log scale - needs to be sign-preserving for negative values + 0
            if apply_symlog : y = symlog(y)
            
            ax.plot(x, y, label = plot_params["label"], color = plot_params["colour"], 
                    linestyle = plot_params["linestyle"], marker = plot_params["marker"], 
                    markersize = plot_params["markersize"] )
        case "kde" : 
            sns.kdeplot(x = y, color = plot_params["colour"], label = plot_params["label"], 
                        linestyle = plot_params["linestyle"], ax = ax, alpha = 0.3, fill = True, common_norm = False)
        case "histplot" : 
            sns.histplot(x = y, color = plot_params["colour"], linestyle = plot_params["linestyle"], 
                        label = plot_params["label"], ax = ax, fill = True, alpha = 0.3, common_norm = False, stat = "probability")
        case _ :
            raise ValueError(f"Invalid plot type for visualisation {plot_params['plot_type']}")

def plot_actexp_figure_data(xvp : expVisual, results_df : pd.DataFrame,
                            xvp_triple : tuple[str, str, str], save2csv : bool = True,
                            nskip : int = 1, verbose : bool = True) -> None :
    
    """
    Given an experiment visual params dataclass, result combination type, plot the figure and all axes and curves upon it.
    
    Params:
        xvp: the experiment visual params object, generated from expConfig.
        
        results_df: dataframe of findings, extracted from main total_activations_df via the xvp triple over
        (eval_type, category, measure_type). Should already be extracted before using this code; do not pass in TAD.
        
        save2csv: whether or not to also save the data as a CSV, since there's no point in making a separate tocsv function.
        
        nskip: how many data examples to skip forward, e.g = 5 means the first 5 examples are skipped. Useful for 
        enforcing visual clarity/stability; first few readings are usually extremely unstable and high-magnitude.
        
    """
    
    # Unpack for ease of use and readability
    eval_type, category, measure_type = xvp_triple
    
    # Delegate main work to dataclass to avoid unnecessary code bloat here
    figure_data = xvp.generate_figure_params(*xvp_triple)
    if save2csv : df2csv(results_df, f"{xvp.save_folder}/csvs/{figure_data['savename']}.csv")
    
    # len(all_tests) should always be <= nrows * ncols by way of construction
    fig, axes = plt.subplots(
        nrows = figure_data["nrows"],
        ncols = figure_data["ncols"],
        figsize = figure_data["figsize"], 
        squeeze = False
    )
    axes = axes.flatten() # Easier to work with
    
    # Every test will be of the form (reducer, kf_reducer) - anything else e.g index is invalid
    all_tests = [col for col in results_df.columns.tolist() if isinstance(col, tuple)]
    reducers = {reducer for reducer, kf_reducer in sorted(all_tests)}
    
    # Each test type gets its own axes object. Had to use set to avoid duplicates,
    # since each test appears len(kf_reducers) times over all the columns
    test2ax = dict(zip(reducers, axes))
    activation_groups = results_df.groupby("activation")
    
    for reducer, kf_reducer in all_tests :
        
        test_agg_str = f"[Reducer: {reducer}, KFold reducer: {kf_reducer}]"
        if verbose : print(f"Visualising configuration: {test_agg_str}")
        
        ax = test2ax[reducer]
        ax_params = xvp.generate_axes_params(reducer, figure_data, nskip = nskip)
        populate_axes(ax, ax_params)
        
        for activation_name in xvp.experiment.activation_names :
            if verbose : print(f"{test_agg_str} Plotting activation {activation_name}")
            activation_data = activation_groups.get_group(activation_name)

            x = ax_params["xaxis"]
            y = activation_data[(reducer, kf_reducer)]
            
            # Must not skip layers, because too few of them 
            if is_ordinal(measure_type) and measure_type != "layers" : 
                x = x[ax_params["nskip"]: ]
                y = y[ax_params["nskip"]: ]
                
            plot_params = xvp.generate_plot_params(activation_name, category, kf_reducer, figure_data["plot_type"])
            # Must not apply symlog on metrics since metrics naturally enforce bounding, and could distort results
            plot_data(x, y, ax, plot_params, apply_symlog = False if category == "metrics" else True)
        
    for ax in axes : 
        post_plotting_axes_ops(ax)
        
    savename = f"{xvp.save_folder}/figures/{figure_data['savename']}.pdf"
    
    fig.suptitle(figure_data["title"], fontsize = "xx-large")
    fig.tight_layout()
    fig.savefig(savename) # do NOT use plt.savefig, it will cause a memory leak due to bug in pyplot
    plt.close(fig)
    
    
def plot_activation_tests(act_results : activationResults, xp : expConfig,
                          verbose : bool = True) -> None :

    """
    Plot all figures from the total results dataframe, given expConfig used to generate said results.
    
    Params:
        results: dictionary of Pandas dataframes, each of which stores results over 
        a given (eval_type, category, measure_type) combination. Each combination requires its own figure.
        
        xp: expConfig object used to generate results, usually via complete_activation_test. This 
        is immediately discarded to create the experimentVisualParams.
        
        verbose: whether to show detailed work-in-progress for plotting.
    
    """

    # In theory could swap expConfig as input arg for xvp directly, but more clear this way
    xvp = xp.exp_vis_params()
    print(f"Save folder created at: {xvp.save_folder}")

    create_path(f"{xvp.save_folder}/figures")
    create_path(f"{xvp.save_folder}/csvs")
    
    with Progress() as progress : 
        work = progress.add_task("Visualisation progress:", total = len(act_results.results) )
        
        for eval_type, category, measure_type in act_results.results :
            progress.console.log(f" -> Visualising {eval_type} results on {category} data measured over {measure_type}.")
            
            total_activation_df = act_results.results[(eval_type, category, measure_type)]
            plot_actexp_figure_data(xvp, total_activation_df, (eval_type, category, measure_type), 
                                    verbose = verbose)
            
            progress.advance(work, 1)