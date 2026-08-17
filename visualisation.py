import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from rich.progress import Progress
from typing import Any
from sklearn.metrics import r2_score

### CUSTOM
from categories.base_definitions import is_ordinal
from dataclass_objects.config_objects import expVisual, expConfig
from dataclass_objects.result_objects import ActivationResults
from support.plotting_helpers import is_empty_axis, df2csv
from support.processing_helpers import symlog, sampling_indices, hash_df, linreg_calc_2d, linreg_calc_3d
from support.parsing_helpers import create_path

matplotlib.use('Agg') # No interactive window, purely file-based rendering
plt.style.use("seaborn-v0_8-whitegrid")

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
    
    ax.legend(fontsize = 9)   
     
    # Show the x-axis for reference
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax : 
        ax.axhline(0, 0, 1, linestyle = "--", color = "red")
    
def plot_data(x : np.ndarray, y : np.ndarray, ax, plot_params : dict[str, Any], z : None | np.ndarray = None, 
              apply_symlog : bool = True ) -> None :
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
        z: optional z-axis data. Must equal length of y. If specified and using "curve" plot type, then the ax must be 3D.
    
    """
    
    is_3d = z is not None
    
    # Switch-case offers only marginal speedup gains over branching, but more readable + extensible
    match ( is_3d, plot_params["plot_type"] ) :
        case (False, "curve" ) :

            # Symmetric log scale - needs to be sign-preserving for negative values + 0
            if apply_symlog : y = symlog(y)
            
            ax.plot(x, y, label = plot_params["label"], color = plot_params["colour"], 
                    linestyle = plot_params["linestyle"], marker = plot_params["marker"], 
                    markersize = plot_params["markersize"], markevery = len(x) // 10 + 1 )
        case (True, "curve" ) :
            print("Warning: 3D curves (planes) not currently supported. Data is 2D and will require regression for 3D.")
            
        case (False, "scatter") :
            
            if apply_symlog : y = symlog(y)
            
            ax.scatter(x, y, label = plot_params["label"], color = plot_params["colour"],
            s = 32, edgecolor = "black") 
            
            (w, b), y_hat, R2 = linreg_calc_2d(x, y)
            
            ax.plot(x, y_hat, label = f"$y(x) = {w:.2f}x + {b:.2f}, R^2 = {R2:.2f}$",
                    color = plot_params["colour"], marker = plot_params["marker"], 
                    markersize = plot_params["markersize"], markevery = len(x) // 10 + 1 )
            

        case (True, "scatter") :
            
            if z is None : return # Pointless None check since is_3d is True, but linter cries otherwise
            if apply_symlog: z = symlog(z)
            
            ax.scatter(x, y, z, label = plot_params["label"], color = plot_params["colour"],
                       s = 32, edgecolor = "black")
            
            (w1, w2, b), (X, Y, Z), R2 = linreg_calc_3d(x, y, z) # Helper function avoids calculations inside plotting func
            
            ax.plot_surface(X, Y, Z, label = f"$Z(X, Y) = {w1:.2f}X + {w2:.2f}Y + {b:.2f}, R^2 = {R2:.2f}$",
                            color = plot_params["colour"], alpha = 0.30)
            
            
        case (False, "kde") : 
            sns.kdeplot(x = y, color = plot_params["colour"], label = plot_params["label"], 
                        linestyle = plot_params["linestyle"], ax = ax, alpha = 0.3, fill = True, common_norm = False)
        
        case (True, "kde") :
            
            sns.kdeplot(x = y, y = z, color = plot_params["colour"], label = plot_params["label"], 
                        linestyle = plot_params["linestyle"], ax = ax, alpha = 0.3, fill = True, common_norm = False )    
        
        case (False, "histplot" ) : 
            sns.histplot(x = y, color = plot_params["colour"], linestyle = plot_params["linestyle"], 
                        label = plot_params["label"], ax = ax, fill = True, alpha = 0.3, common_norm = False, 
                        stat = "probability")
        
        case (True, "histplot") :
            sns.histplot(x = y, y = z, color = plot_params["colour"], linestyle = plot_params["linestyle"], 
                        label = plot_params["label"], ax = ax, fill = True, alpha = 0.3, common_norm = False, 
                        stat = "probability")
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
    figure_data = xvp.generate_figure_params(*xvp_triple, apply_symlog = category != "metrics")
    if save2csv : df2csv(results_df, f"{figure_data['savename']}.csv", f"{xvp.save_folder}/csvs" )
    
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
            y = activation_data[(reducer, kf_reducer)].to_numpy()
            
            # Must not skip layers, because too few of them 
            if is_ordinal(measure_type) and measure_type != "layers" : 
                x = x[ax_params["nskip"]: ]
                y = y[ax_params["nskip"]: ]
                
            plot_params = xvp.generate_plot_params(activation_name, category, kf_reducer, figure_data["plot_type"])
            # Must not apply symlog on metrics since metrics naturally enforce bounding, and could distort results
            plot_data(x, y, ax, plot_params, apply_symlog = figure_data["apply_symlog"])
        
    for ax in axes : 
        post_plotting_axes_ops(ax)
        
    savename = f"{xvp.save_folder}/figures/{figure_data['savename']}.pdf"
    
    fig.suptitle(figure_data["title"], fontsize = "xx-large")
    fig.tight_layout()
    fig.savefig(savename) # do NOT use plt.savefig, it will cause a memory leak due to bug in pyplot
    plt.close(fig)
    
    
def plot_activation_tests(act_results : ActivationResults, xp : expConfig, verbose : bool = False) -> None :

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
    xvp = xp.generate_expvisual()
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
            

def plot_time2threshold_tests(thresh_results : dict[tuple[str, str, str], pd.DataFrame], xp : expConfig, 
                              save2csv : bool = True) -> None :
    
    """Generates every possible valid plot of time-to-threshold data from a thresh_results dictionary, given expConfig
    used to generate those results. 
    
    Params:
        thresh_results: the dictionary mapping (eval_type, category_type, reducer_type) to a results DataFrame.
        xp: the original expConfig dataclass used to generate those results.
        save2csv: whether to save the results to CSV files or not.
    """
    
    print("Visualising threshold result data...")
    exp_vis = xp.generate_expvisual()
    create_path(f"{exp_vis.save_folder}/figures/threshold_tests")
    create_path(f"{exp_vis.save_folder}/csvs/threshold_tests")

    with Progress() as progress : 
        work = progress.add_task("Threshold visualisation progress:", total = len(thresh_results))
            
        for (eval_type, cat, red), thresh_df in thresh_results.items() :
            
            progress.console.log(f" -> Visualising {eval_type} threshold results on {cat} data with reducer {red}.")
            
            plot_time2threshold_test(thresh_df, exp_vis, eval_type, cat, red, save2csv)
            progress.advance(work, 1)
                
            
def plot_time2threshold_test(thresh_df : pd.DataFrame, exp_vis : expVisual, 
                              eval_type : str, category_name : str, reducer_name : str, save2csv : bool = True) -> None :
    
    """Plot the results from a time-to-threshold test with the specified parameters.
    
    Params:
        thresh_df: the DataFrame that stores the results; get this from time2threshold_test(). 
        exp_vis: the visual parameters dataclass for the experiment. If you have the object, use xp.generate_expvisual().
        eval_type: whether the data is train or test data.
        category_name: the name of the category type, e.g "testloss" or "aouts".
        reducer_name: the aggregator function that marginalises all other dimensions into the epoch dimension.
        save2csv: whether to save the data as a CSV or not.
    """
    
    print(f"Plotting time-to-threshold {eval_type} data for category {category_name} and reducer type {reducer_name}...")
    filename = f"threshtest-{eval_type}data-{category_name}-{reducer_name}"
    if save2csv : df2csv(thresh_df, filename + ".csv", f"{exp_vis.save_folder}/csvs/threshold_tests")
    
    fig_params = exp_vis.generate_figure_params(eval_type, category_name, "epochs", apply_symlog = False)
    ax_params = exp_vis.generate_axes_params(reducer_name, fig_params )
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = fig_params["figsize"], squeeze = True)
    
    # Matplotlib normally puts lowest x values on the left and highest ones on the right; if descending, need to fix this
    ascending = thresh_df.index[-1] >= thresh_df.index[0]
    xaxis = thresh_df.index.to_numpy()
    for activation_name in thresh_df.columns.tolist() :

        plot_params = exp_vis.generate_plot_params(activation_name, category_name, "mean", "curve")
        dataseries = thresh_df[activation_name].to_numpy()
        plot_data(xaxis, dataseries, ax, plot_params, apply_symlog = fig_params["apply_symlog"])
    
    if not ascending : ax.invert_xaxis() # Fix for non-ascending data
    ax.grid(ax_params["grid"])
    ax.set_xlabel(f"{category_name.title()} {reducer_name} threshold value", fontsize = "x-large")
    ax.set_ylabel("# epochs to reach threshold", fontsize = "x-large")
    plt.title(f"Time-to-threshold (T2T) test: {fig_params['title']} ", fontsize = "xx-large")
    post_plotting_axes_ops(ax)
    
    fig.tight_layout()
    create_path(f"{exp_vis.save_folder}/figures/threshold_tests")
    fig.savefig(f"{exp_vis.save_folder}/figures/threshold_tests/{filename}.pdf") # do NOT use plt.savefig
    plt.close(fig)
    

def plot_df_features(coord_df : pd.DataFrame, xpc : expConfig, plot_type : str = "scatter", 
                     save2csv : bool = True) -> None :
    
    """Associated visualiser function for activationResults.coords2features() and .coord2feature() from the 
    activationResults custom dataclass in dataclass_objects/result_objects.py. As such, expects a Pandas
    Multi-indexed DataFrame of 2 index levels; the activation and the coordinate, with no restriction on
    coordinate names. Can plot for any pair or triple of features given, but not more than 3 coordinates.
    If more than 3 given, uses the first 3. If only 1 given, uses a KDE. Accepts all plot types except for
    "curve" in 3D (2D supported).
    
    As it makes use of activationResults, requires both an instance of complete_activation_test() or its sub-cases
    to generate the dataclass, followed by using one or both of the above methods. Will attempt to visualise all 
    features stored in the coord_df, so please make sure to select the right coordinate types before using this
    visualiser function, or you may not get the result you want.
    
    Results from this function may be found in (experiment folder name)/figures/features/* for figures, and 
    (foldername)/csvs/features/* for CSV data.
    
    Params:
        coord_df: the coordinate DataFrame generated from the coordinate Series component of an activationResults df.
        xpc: the configuration used to run the original experiment.
        plot_type: the type of plot. Supports "curve" (2d only), "scatter", "kde" and "histplot". Scatter supports regression.
        save2csv: whether to save the results as a CSV or not.
        
    Raises:
        ValueError: No coordinates provided.
    """
    
    xvp = xpc.generate_expvisual() 
    all_activations : list[str] = coord_df.columns.get_level_values(0).unique().tolist()
    all_coords : list[str] = coord_df.columns.get_level_values(1).unique().tolist()
    
                                                        # "_" is my face when having to write hash code
    df_name = "-".join([act_name[0] for act_name in all_activations]) + "_" + hash_df(coord_df, maxlen = 10)
    if save2csv : df2csv(coord_df, foldername = f"{xvp.save_folder}/csvs/features", filename = f"{df_name}.csv" )
    
    print(f"Plotting features over activations {", ".join(all_activations)} in save folder {xvp.save_folder}...")
    
    if len(all_coords) > 3 :
        print(f"Warning: Attempted to plot {len(all_coords)} features. This is not visualisable. Using first 3.")
    elif len(all_coords) <= 0 :
        raise ValueError("No coordinates provided, or df features not of multiindex type. Should be of form (activation, coordinate)")   

    
    is_1d = len(all_coords) == 1
    if is_1d and plot_type not in ["kde", "histplot"] :
        print("Warning: 1D data cannot be of the type \"curve\". Switching to kde")
        plot_type = "kde"
        
        
    is_3d = len(all_coords) >= 3 
    use_3d_axes = is_3d and plot_type == "scatter" # 3D KDEs and Histplots use Seaborn 2D axes; only scatter needs surfaces.
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (1920/96, 1080/96), 
                           subplot_kw = {"projection" : "3d" } if use_3d_axes else {} )

    # All activations have the same coordinate dimensions, so just pick the first one to sample
    coord_names = coord_df[all_activations[0]].columns.tolist()

    for activation in all_activations :
                
        activation_coords = coord_df[activation] # Exact category is irrelevant here, so "grad" as placeholder
        plot_params = xvp.generate_plot_params(activation, "grad", "mean", plot_type)
        
        print(coord_names)
        print(activation_coords.columns)
        
        # Must be strictly NumPy arrays or the regression code can fail
        x = activation_coords[coord_names[0]].to_numpy()
        y = activation_coords[coord_names[1]].to_numpy()
        z = activation_coords[coord_names[2]].to_numpy() if is_3d else None
        
        if not is_1d :
            plot_data(x, y, ax, plot_params, z, apply_symlog = True)
        else : # First axes is unused if using kde / histplot
            plot_data(np.array(["Nobody: a film by Saul Goodman"]), x, ax, plot_params, None, apply_symlog = True)

    ax.grid(True)
    ax.set_xlabel(f"{coord_names[0]}", fontsize = "x-large")
    if not is_1d : ax.set_ylabel(f"{coord_names[1]}", fontsize = "x-large")
    if isinstance(ax, Axes3D) : ax.set_zlabel(f"{coord_names[2]}",  fontsize = "x-large")
    
    plt.suptitle(f"Comparing activation features for activations {', '.join(all_activations)}", fontsize = "xx-large")
    
    fig.tight_layout()
    ax.legend()
    create_path(f"{xvp.save_folder}/figures/features")
    fig.savefig(f"{xvp.save_folder}/figures/features/{df_name}.pdf") # do NOT use plt.savefig
    plt.close(fig)