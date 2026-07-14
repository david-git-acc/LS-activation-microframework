from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from typing import Any, Callable
import torch
from torch import nn
import matplotlib.lines as mlines
import json
import hashlib
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
import copy

# CUSTOM
from networks import ActivationNetwork
from support.parsing_helpers import validate_activation_df_column_names, is_hashable, smart_str, safe_asdict, get_name, extract_tuple_list
from support.processing_helpers import sampling_indices, params2grad_vector, pad_torch_stack
from support.plotting_helpers import get_n_colours, determine_plot_type, generate_plot_title
from support.torch_reducers import arithmetic_mean

################################# REST ########################################################

@dataclass
class expConfig() :
    
    """
    Main dataclass for conducting activation experiments efficiently. This dataclass is designed
    to compare the effectiveness of different activation functions on a given network using a set of 
    test functions (reducers) to marginalise unwanted dimensions and compare 1D arrays of results.
    
    Params:
        df_train: dataframe for train data.
        df_test: dataframe for test data.
        
        labels: list of columns that are to be predicted from features (presumed rest)
        
        network_type: type of neural network to work with for the experiment and perform predictions.
        
        loss: loss metric to judge predictions by. 
        
        feature_transforms: tuple of tuples, where the first entry of each tuple is the list of columns to be transformed,
        and the second entry is the transformer class to apply the transformation to each of the columns.
        
        label_transforms: same as feature_transforms, but for labels.
        
        reducers: tuple of test functions to apply on finished results to marginalise unwanted dimensions. 
        activations: tuple of all activation functions to run in the experiment.
        
        kf_reducers: tuple of all aggregation functions to collapse folds over in K-fold crossvalidation, 
        e.g mean, variance, etc over folds. Note for test data it's interpreted as 1-fold crossvalidation, and only 
        mean is permitted for this.
        
        lr: constant learning rate. Can be changed to a variable one.
        
        kfold_k: number of folds to use in Kfold cross validation. 10 is the industry standard.
        
        n_alphas: number of different alpha values in the range (0,1) to use for LS sensitivity testing, with uniform spacing.
        E.g n_alphas = 5 would imply the alpha values (0.0, 0.25, 0.50, 0.75, 1.0).
        
        batch_size: size of train set used per iteration per epoch. Higher values increase gradient accuracy at the cost of time.
        
        max_samples: maximum number of samples to record (does not change number of epochs). Higher values create denser graphs.
        
        epochs: number of total training runs to apply to the neural network, where each epoch is a full pass of df_train.
        
        activation_names: list of display names for the activation functions, index-linked with activations.

        reducer_names: list of display names for the test functions. Again, index-linked with reducers.
        
        kf_reducer_names: list of display names for the aggregation functions, index-linked with kf_reducers.
        
        features_dtype: datatype of all features. Multiple datatypes for different features are not currently supported.
        
        labels_dtype: same as features_dtype but for labels.
        
        categories: tuple of all categories to test over. Defaults to all of them if none selected.

    """
    
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    labels : str | list[str]
    network_type : type[ActivationNetwork]
    loss : nn.Module
    feature_transforms : tuple[tuple[list[str], Any]]
    label_transforms : tuple[tuple[list[str], Any]]
    activations : list[nn.Module] | tuple[nn.Module, ...]
    reducers : tuple[Callable, ...]
    kf_reducers : tuple[Callable, ...]
    lr : float = 0.001
    kfold_k : int = 10
    n_alphas : int = 5
    batch_size : int = -1 # full batch
    max_samples : int = -1 # no limit
    epochs : int = 500
    activation_names : list[str] = field(default_factory = list)
    reducer_names : list[str] = field(default_factory = list)
    kf_reducer_names : list[str] = field(default_factory = list)
    features_dtype : torch.dtype = torch.float32
    labels_dtype : torch.dtype = torch.long
    categories : tuple[str, ...] | str = ("grad",)
    
    def __post_init__(self) :
        
        """
        Validation of all inputs to make sure the experiment runs smoothly and every activation has a valid name.
        """
        
        # YAML doesn't support tuples so convert in the program
        for potentially_list_attribute in ( "activations", "reducers", "kf_reducers", "categories" ) :
            attr = getattr(self, potentially_list_attribute)
            if isinstance(attr, list) :
                setattr(self, potentially_list_attribute, tuple(attr))
        
        # Placate the linter + consistency
        if isinstance(self.labels, str) : 
            self.labels = [self.labels]
        
        # Same as above
        if isinstance(self.categories, str) :
            self.categories = (self.categories, )

        self.reducer_names = validate_activation_df_column_names(self.reducers, self.reducer_names)
        self.activation_names = validate_activation_df_column_names(self.activations, self.activation_names)
        self.kf_reducer_names = validate_activation_df_column_names(self.kf_reducers, self.kf_reducer_names)
        
        # Mean is non-optional for expConfig, we need it for when we collect results over test data
        if "arithmetic_mean" not in {get_name(f) for f in self.kf_reducers } : # Use this since funcs have no ==
            self.kf_reducer_names.append("mean")
            self.kf_reducers += (arithmetic_mean, )
                
    def savename(self , maxlen : int = 10) -> str :
        
        """Generates a deterministic savename for each experiment based on its deterministically string-able properties.
            For regular experiments, shows the network name, the initials of all activations used, then the signature hash.
            For LS alpha sensitivity experiments, shows the network name, the activation, number of alphas, and the hash.
            *Any* change in parameters or potentially the code structure itself will likely result in a different hash.

        Params:
            maxlen: defaults to 10, maximum length of the hash component.

        Returns:
            The savename for the experiment. Useful to track experiments and avoid experiment crowding in the main directory.
        """
        
        # Salt added on end
        strformat = str([ smart_str(v) if is_hashable(v) else smart_str(get_name(v))
                         for v in self.__dict__.values() ])
        as_str = json.dumps(strformat + "you and I, (nothing comes easy) * 2", sort_keys = True).encode() 
        
        hash_monstrosity = hashlib.sha256(as_str).hexdigest()
        
        if len(self.activation_names) == 1 :
            plus_extra = f"{self.activation_names[0]}_alpha[{self.n_alphas}]"
        else :
            plus_extra = "_".join([name[0] for name in self.activation_names])
        
        readable_metadata = f"{self.network_type.__name__}-{plus_extra}"
        
        return "exp-" + readable_metadata + "-" + hash_monstrosity[:maxlen] 
    
    def exp_vis_params(self) -> expVisual :
        
        """
        Generates the visual parameters dataclass for this activation function. 

        Returns:
            The visual parameters dataclass, with its associated attributes and methods.
        """
        
        main_params = {
            "save_folder" : self.savename(),
            "activation_colours" : dict(zip(self.activation_names, 
                                            get_n_colours(len(self.activations)) )),
            "kf_aggfunc_linestyles" : dict(zip(self.kf_reducer_names, 
                                               list(mlines.lineStyles.keys())[:len(self.kf_reducers)])),
            "experiment" : self,
        }
        
        return expVisual(**main_params)
    
    def exp_inp_params(self) -> expInput :
        return expInput(**safe_asdict(self.__dict__, expInput))
    
@dataclass
class expVisual() :
    
    """
    This is the visual parameters dataclass for a given experiment. 
    
    Params:
        save_folder: save folder name for the experiment, generated by experiment.savename().
        activation_colours: different colours for each activation function for differentiation.
        kf_aggfunc_linestyles: different linestyles per kfold aggregation function for differentiation.
        experiment: reference to the original experiment object that created it.
        
    """
    
    save_folder : str
    activation_colours : dict[str, Any]
    kf_aggfunc_linestyles : dict[str, str]
    experiment : expConfig
    
    def generate_figure_params(self, eval_type : str, category : str, measure_type : str,
                               plots_per_row : int = 3) -> dict[str, Any] :
        
        """
        Generate parameters dictionary for a given figure and triple (eval_type, category, measure_type).
        
        Params:
            eval_type: the evaluation type (train/test) of the data. 
            category: what type of data (gradient, testloss, test predictions, etc) is being measured.
            measure_type: what the valid dimension (independent variable) is. Usually "epochs", but not always.
            plots_per_row: how many plots should be represented on each given row. Defaults to 3 for appearances.

        Returns:
            dictionary of parameters for the given figure, to be plugged in during visualisation.
        """
        
        nrows = ceil(len(self.experiment.reducers) / plots_per_row)
        ncols = min(plots_per_row, len(self.experiment.reducers))
        
        # For testloss there are no tests we can perform, so will always be exactly 1 figure even if other tests exist
        special_cases = { "testloss" : (1, 1) }
        if category in special_cases : nrows, ncols = special_cases[category]
        
        title = generate_plot_title(category, 1 if eval_type == "test" else self.experiment.kfold_k)
        plot_type = determine_plot_type(eval_type, category, measure_type)
        
        # All curves get symlog treatment for numerical stability
        if plot_type == "curve" : title += " (symlog)"
        
        fig_params = {
            "savename" : f"{category}-{measure_type}_on_{eval_type}",
            "figsize" : (1920/96, 1080/96),
            "eval_type" : eval_type,
            "category" : category,
            "measure_type" : measure_type,
            "title" :  title,
            "plot_type" : plot_type,
            "nrows" : nrows,
            "ncols" :  ncols,
            "nplots" : nrows * ncols           
        }
        
        return fig_params
    
    def generate_axes_params(self, reducer : str, fig_params : dict[str, Any], nskip: int = 5) -> dict[str, Any] :
        
        """Generates parameters for the given axes object on a given figure.
        
        Params:
            reducer: what test function the axes is for. Every axes object is for a specific test function.
            
            fig_params: the associated dictionary of parameters for the parent figure object. Use generate_figure_params()
            if this is not available from the same dataclass object expVisual.
            
            nskip: Number of initial epochs to skip. Only valid for epochs or ordered x-axes. 

        Returns:
            Dictionary of parameters for the given axes object.
        """
        
        match fig_params["plot_type"] :
            case "curve" :  # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"
                xlabel, ylabel = (fig_params["measure_type"][:-1], reducer) 
            case "kde" | "histplot" :
                xlabel, ylabel = (reducer, "frequency density")
            case _ :
                xlabel, ylabel = ("x-axis placeholder label", "y-axis placeholder label")

        xticklabels = sampling_indices(self.experiment.epochs, self.experiment.max_samples)
        
        ax_params = { 
            "reducer" : reducer, 
            "plot_type" : fig_params["plot_type"],
            "xlabel" : xlabel,
            "ylabel" : ylabel,
            "grid" : True, 
            "nxticks" : 10,
            "xticklabels" : xticklabels,
            "nskip" : nskip
        }
    
        return ax_params

    def generate_plot_params(self, activation_name : str, kf_reducer : str, plot_type : str) :
        
        """
        Generates parameters for the given plot object for an axes object (axes itself not required).
        
        Params:
            activation_name: the name of the activation to plot over (determines colour).
            kf_reducer: the type of aggregation function used (determines linestyle).
            plot_type: the type of plot (kde, curve, histplot, etc).

        Returns:
            The dictionary of associated parameters for the axes object.
        """
        
        plot_params = {
            "activation_name" : activation_name,
            "kf_reducer" : kf_reducer,
            "plot_type" : plot_type,
            "label" : f"fold-{kf_reducer}({activation_name})",
            "colour" : self.activation_colours[activation_name],
            "linestyle" : self.kf_aggfunc_linestyles[kf_reducer],
            "markersize" : 4.0,
            "marker" : "^"
        }
        
        return plot_params
    
@dataclass
class expInput() :
    X_train_tensor : torch.Tensor
    X_test_tensor : torch.Tensor
    Y_train_tensor : torch.Tensor
    Y_test_tensor : torch.Tensor 
    anet_model : ActivationNetwork
    target_loss : nn.Module = nn.CrossEntropyLoss()
    optim_type : type[torch.optim.Adam] | type[torch.optim.SGD] = torch.optim.Adam
    epochs : int = 500 
    lr : float = 0.001
    batch_size : int = -1
    max_samples : int = -1
    categories : tuple[str, ...] = ("grad",)
    
    def __post_init__(self) : 
        # Use full-batch GD if no batch size given
        self.batch_size = len(self.X_train_tensor) if self.batch_size == -1 else self.batch_size
        self.max_samples = self.epochs if self.max_samples == -1 else self.max_samples
        
        self.training_dataset = TensorDataset(self.X_train_tensor, self.Y_train_tensor)
        self.training_dataloader = DataLoader(self.training_dataset, self.batch_size, shuffle = True )
        
        self.nabla_shape = params2grad_vector(self.anet_model.parameters()).size()
        self.optim = self.optim_type(self.anet_model.parameters(), lr = self.lr)
        self.saved_params : dict[str, Any] = {}
        
    
    def save_state(self) -> None :
        self.saved_params["anet_model"] = { param : value.clone() 
                                           for param, value in self.anet_model.state_dict().items() }
        self.saved_params["optim"] = copy.deepcopy(self.optim.state_dict())
        
    
    def reload_state(self) -> None :
        self.anet_model.load_state_dict(self.saved_params["anet_model"])
        self.optim.load_state_dict(self.saved_params["optim"])
    
    @property
    def n_captures(self) -> int :
        return min(self.epochs, self.max_samples)
    
    
@dataclass
class testInput() :
    X : torch.Tensor
    reducers : tuple[Callable, ...]
    reducer_names : str | list[str]
    kf_reducers :  tuple[Callable, ...]
    kf_reducer_names : str | list[str]
    expected_ndims : int = 2
    measure_type : str = "epochs"
    metadata : dict[str, Any] = field(default_factory = dict, init = True)
    xpc : expConfig | None = None
    
    def __post_init__(self) :
      
        if not isinstance(self.kf_reducers, tuple) : self.kf_reducers = ( self.kf_reducers, )  
        if not isinstance(self.reducer_names, list) : self.reducer_names = [self.reducer_names]
        if not isinstance(self.kf_reducer_names, list) : self.kf_reducer_names = [self.kf_reducer_names]
        
        is_test_data = len(self.X.size()) == self.expected_ndims
        
        # If it's test data then there are no folds, so to avoid having to duplicate this function we add a dummy one
        if is_test_data :
            self.X = self.X[..., None]
        
        self.reducer_names = validate_activation_df_column_names(self.reducers, self.reducer_names)
        self.kf_reducer_names = validate_activation_df_column_names(self.kf_reducers, self.kf_reducer_names)

@dataclass
class experimentResult() :
    
    """
    Simple container class for efficiently representing all categories of result from an experiment. 
        Not intended for any complex calculations, unlike expConfig or expVisual.
    
    Params:
        _results: the dictionary of results, where each key is a category and the value is the tensor of results.
        results: same as _results, stored for type checking and mypy purposes. No need to pass in any value here.
    """
    _results : dict[str, torch.Tensor] | list[experimentResult] = field(default_factory = dict)
    results : dict[str, torch.Tensor] = field(default_factory = dict, init = False) # Should NOT be writeable to
    metadata : dict[str, Any] = field(default_factory = dict, init = True)

    def __post_init__(self) :

        # Handle k-fold assumption
        if isinstance(self._results, list) :
            
            # Construct new dictionary to store each one
            dict_results = {}
            
            for exp_result in self._results : 
                
                for category, result in list(exp_result.results.items()) :              
                    if category in dict_results :
                        dict_results[category].append(result)
                    else :
                        dict_results[category] = [result]
            
            # Creates the K-fold architecture for the results. Setting dim = -1 sets kfold dim as last one (required)  
            # pad_torch_stack had to be specifically developed for SKF not having equal fold sizes, rest shouldn't need it       
            dict_results = {category : torch.stack(pad_torch_stack(data), dim = -1) 
                            for category, data in dict_results.items()}
            self.results = dict_results
            
        else :
            self.results = self._results

    def get_ndims(self) -> dict[str, int] :

        ndims = { name : len(x.size()) for name, x in self.results.items() if isinstance(x, torch.Tensor)}
        
        return ndims
    
    def get_ndims_tuple(self) -> tuple[tuple[str, int], ...] :
        
        ndims = tuple( ( name , len(x.size()) ) for name, x in self.results.items() if isinstance(x, torch.Tensor) )
        
        return ndims
    
    def get_max_dim(self) -> int :
        
        return max(self.get_ndims_tuple(), key = lambda x : x[1])[1]


@dataclass
class activationResults() :
    results : dict[tuple[str, str, str], pd.DataFrame]
    
    def __post_init__(self) :
        self.figure_coord_types : tuple[str, ...] = ("eval_type", "category", "measure_type",)
        self.df_coord_types : tuple[str, ...] = ("reducer", "kf_reducer")
        self.activation_coord_type : str = "activation"
        self.coordinate_types : tuple[str, ...] = self.figure_coord_types + self.df_coord_types + (self.activation_coord_type, )
    
        accumulated_dfs = []
        for figure_coords, df in self.results.items() :
            
            # Don't want to change original results data, may be reused for other purposes
            new_df = df.copy()
            
            # Adds all the identifiers for the figure directly into the dataframe for identification, no more dict structure
            for figure_coord_type, figure_coord in list(zip(self.figure_coord_types, figure_coords)) :
                new_df[figure_coord_type] = figure_coord
                
            # Need to know how to represent the data in order or we will get a jumbled mess at the end
            new_df.reset_index(names = ["position"], drop = False, inplace = True)    
            all_other_columns = [col for col in new_df.columns if not isinstance(col, tuple)]
            
            # This turns the tuple columns into values using the var name "agg-test-type", needed to separate. 
            new_df_melted = new_df.melt(id_vars = all_other_columns, var_name = "agg-test-type", value_name = "val" )

            # But now need to separate agg and test type since still as tuples; do this by turning to list of tuples of len 2
            test_kf_reducer_tuples = new_df_melted["agg-test-type"].tolist()

            # Provides the agg-test-type columns
            new_df_melted[list(self.df_coord_types)] = pd.DataFrame(test_kf_reducer_tuples, index = new_df_melted.index)
            new_df_melted.drop(columns = ["agg-test-type"] , inplace = True)
            
            accumulated_dfs.append(new_df_melted)
            
        accumulated_df : pd.DataFrame = pd.concat(accumulated_dfs, axis = 0, ignore_index = True)
        reordered_columns = list(self.coordinate_types) + ["position", "val"] # Honours the order given in the class
        
        # Val should always be the last entry
        self.df : pd.DataFrame = accumulated_df[reordered_columns]
        

    def query(self, eval_type : str | None = None, category : str | None = None, measure_type : str | None = None, 
              reducer : str  | None = None, kf_reducer : str  | None = None, activation : str  | None = None) -> pd.DataFrame :
        
        
        query_requirements = [eval_type, category, measure_type, reducer, kf_reducer, activation]
        coordinate_valuations = list(zip(list(self.coordinate_types), query_requirements))
        
        # Begin with the all-true mask then apply each condition to filter out irrelevant tuples in the search     
        query_mask = pd.Series(True, index = self.df.index)
        
        # Build up each condition, removing all which do not comply to get us our line data
        for coordinate_type, coordinate in coordinate_valuations :
            # If user does not specify, just return all possible valuations === no condition, tautological condition
            if coordinate is None : continue
            
            condition = self.df[coordinate_type] == coordinate
            query_mask = query_mask & condition # All conditions must be met for it to qualify under our query
                        
        # Sort by position to make sure the data remains in the correct ordering; integrity. 
        # Most likely was always in the right order anyway but this guarantees it
        query_result = self.df[query_mask].sort_values(by = "position", ascending = True)
  
        # Want only val, not categories or position
        return query_result
        
    def specific_query(self, eval_type : str, category : str, measure_type : str, 
                       reducer : str, kf_reducer : str, activation : str, replace_index : bool = True) :
    
        query_requirements = [eval_type, category, measure_type, reducer, kf_reducer, activation]
        
        if None in query_requirements :
            raise ValueError(f"NoneType parameter given for specific query {query_requirements}")
        
        query_result = self.query(*query_requirements)

        # May not want to keep "position"
        if replace_index :
            query_result.index = query_result["position"]
            query_result.index.name = category[:-1] # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"

        # Since all coordinates will be identical, no point in keeping the exact coords
        return query_result[["val"]]
