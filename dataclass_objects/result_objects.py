from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from typing import Any
import torch

# CUSTOM
from support.parsing_helpers import singularise
from support.processing_helpers import pad_torch_stack, nan_long

@dataclass
class ExperimentResult() :
    
    """
    Simple container class for efficiently representing all categories of result from an experiment. 
        Not intended for any complex calculations, unlike expConfig or expVisual.
    
    Params:
        _results: the dictionary of results, where each key is a category and the value is the tensor of results.
        NOTE: If passing in a list of experimentResults, will apply torch.stack and concatenate on the last dimension.
        results: same as _results, stored for type checking and mypy purposes. No need to pass in any value here.
    """
    _results : dict[str, torch.Tensor] | list[ExperimentResult] = field(default_factory = dict)
    results : dict[str, torch.Tensor] = field(default_factory = dict, init = False) # Should NOT be writeable to
    metadata : dict[str, Any] = field(default_factory = dict, init = True)

    def __post_init__(self) :

        self.instantiate_results()

    def instantiate_results(self) -> None :
        
        """Create the self.results dictionary object for the ExperimentResult.
        """
        
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
            dict_results = {category : torch.stack(pad_torch_stack(data, 
                            pad_with = nan_long if data[0].dtype == torch.long else torch.nan), dim = -1) 
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
class ActivationResults() :
    
    """Results dataclass for a complete_activation_test(), or related function. 

    Params:
        results: dictionary of triples (eval_type, category, measure_type) which uniquely define a figure key, with the 
        corresponding DataFrame as its value object.
        
        df: implicit attribute calculated during instantiation from results. Represents all data using a 7-coordinate system:
            1. The figure key identifiers (eval_type, category, measure_type)
            2. The axes identifier (reducer)
            3. The specific plot identifier (activation and kf_reducer)
            4. Datapoint ID (position)
        
        All 7 coordinates combined represent exactly 1 datapoint in a given axes object belonging to a figure.

    """
    
    results : dict[tuple[str, str, str], pd.DataFrame]
    
    def __post_init__(self) :
        self.figure_coord_types : tuple[str, ...] = ("eval_type", "category", "measure_type",)
        self.df_coord_types : tuple[str, ...] = ("reducer", "kf_reducer")
        self.activation_coord_type : str = "activation"
        self.coordinate_types : tuple[str, ...] = self.figure_coord_types + self.df_coord_types + (self.activation_coord_type, )
    
        self.rebuild_df()
        
    def rebuild_df(self) -> None :
        
        """Build the DataFrame for ActivationResults dataclass object. 
        """
        
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
            new_df_melted.drop(columns = ["agg-test-type"], inplace = True)
            
            accumulated_dfs.append(new_df_melted)
            
        accumulated_df : pd.DataFrame = pd.concat(accumulated_dfs, axis = 0, ignore_index = True)
        reordered_columns = list(self.coordinate_types) + ["position", "val"] # Honours the order given in the class
        
        # Val should always be the last entry
        self.df : pd.DataFrame = accumulated_df[reordered_columns]
        

    def select(self, eval_type : str | None = None, category : str | None = None, measure_type : str | None = None, 
              reducer : str  | None = None, kf_reducer : str  | None = None, activation : str  | None = None) -> pd.DataFrame :
        
        """Given all 6 possible coordinate types (excluding "position"), filter the results dataframe for all data that 
        satisfies the criteria and output as a new DataFrame object. Not to be confused with results.df.query().
        
        Leaving any coordinate type as None will return all existing valuations as row elements.
        
        Always outputs a DataFrame, not a Series. Will always be 8 columns, one for each coordinate + the value, even if all 
        columns specified. If this behaviour is not desired, consider self.select_specific().
        
        Columns are in the order of:
            1. eval_type: whether the data is KFold train data or multiseed test data.
            2. category: the type of data - e.g activation gradients, LS, or test predictions.
            3. measure_type: which dimension to preserve (e.g epochs, layers, params)
            4. reducer: the aggregator function over all other non-kfold dimensions.
            5. kf_reducer: the KFold aggregator function over the folds.
            6. activation: the corresponding activation name the data corresponds to. 
            7.* position: the index of the data, since the first 5 will always produce a Series.
            8.* val: the actual value assigned to this set of 5 coordinates at the given position.
    
        Returns:
            DataFrame: the desired DataFrame object containing all results after projection.
        """
        
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
        
    def select_specific(self, eval_type : str, category : str, measure_type : str, 
                       reducer : str, kf_reducer : str, activation : str, replace_index : bool = True) -> pd.DataFrame :
    
        """Same as ActivationResults.select(), but returns a single-column Pandas DataFrame. 
        Does not accept NoneType coordinate arguments unlike select(). Unlike select(), will not
        retain other columns in the output DataFrame. 
        
        Params:
            *coordinates: the 6-coordinates to specify. Does not accept "position" as an argument.
            replace_index: whether to keep the "position" index of the output DataFrame intact or not.
            If not, replaces it with the category type.
        
        Returns:
            DataFrame: single-column DataFrame containing value and position. Note that with all 6
            parameters specified, this DataFrame corresponds exactly to a given single-plot on an axes object
            from visualisation.py. 
            
        """    
    
        query_requirements = [eval_type, category, measure_type, reducer, kf_reducer, activation]
        
        if None in query_requirements :
            raise ValueError(f"NoneType parameter given for specific select {query_requirements}")
        
        query_result = self.select(*query_requirements)

        # May not want to keep "position"
        if replace_index :
            query_result.index = query_result["position"]
            query_result.index.name = singularise(category) # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"

        # Since all coordinates will be identical, no point in keeping the exact coords
        return query_result[["val"]].copy() # Avoid settingWithCopy warning


    def coord2feature(self, coordinate : tuple[str, str, str], activation : str, 
                      measure_type : str = "epochs", preserve_coord_orgname : bool = False) -> pd.DataFrame : 
        
        """Create a feature for a given coordinate, activation and measure type. Similar to coords2features,
        but works on exactly 1 coordinate only and also requires a specific activation. If this behaviour is not
        desired, consider a singleton coordinate list for self.coords2features().
        
        Measure type and activation were kept separate to avoid user having to separate inputs between function calls.
        
        Params:
            coordinate: the eval type, category and reducer to turn into a coordinate.
            activation: the name of the specific activation to turn into a feature.
            measure_type: the measure type to use.

        Returns:
            pd.DataFrame: the single-column feature dataframe.
        """
        
        eval_type, category, reducer = coordinate
        coord_query = self.select_specific(eval_type, category, measure_type, reducer, "mean", activation)
        coord_query.sort_index(inplace = True, ascending = True)
        
        coord_name = str(coordinate) if preserve_coord_orgname else "x"
        # Preserve columns in multiindex format despite only 1 coord for code consistency in plotting later
        coord_query.columns = pd.MultiIndex.from_tuples([(activation, coord_name)], names = ("activation", "coord"))
        
        return coord_query
    

    def coords2features(self, coordinates : list[tuple[str, str, str]], measure_type : str = "epochs",
                        preserve_coord_orgnames : bool = False) -> pd.DataFrame :
        
        """Compare 2 or more coordinates of type (eval_type, category_type, reducer_type) with a constant
        kf_reducer (mean) and shared measure_type and convert to features. Useful for comparing results between unrelated
        categories and experiments. 
        
        Converts each coordinate type into its own dimension x_1, x_2, ..., x_n. Therefore can use this to build
        visual and predictive models to compare results that were not directly compared in general functions like 
        complete_activation_test without having to write additional code.
        
        Params:
            coordinates: list of tuples where each tuple is of form (eval_type, category, measure_type).
            measure_type: the shared measure type. Must be shared for proper axes comparison.
            preserve_coord_orgnames: whether to preserve the coordinate names as tuples or rename them to x_i format.

        # NOTE: For visualisation, it will use the first coordinate as x-axis, second as y and third as z. So if you
        want a specific coordinate to be the independent or dependent variable, order it accordingly in the coords list.

        Returns:
            pd.DataFrame: the DataFrame containing results as a Pandas Multiindex DataFrame and corresponding values. Columns are in the multiindex format (activation, coordinate_name).
        """
        
        
        comparison_dict = {}
        for coord_dim, (eval_type, category, reducer) in enumerate( coordinates, start = 1) :
            
            coord_query = self.select(eval_type, category, measure_type, reducer, kf_reducer = "mean")
            
            filtered = coord_query[["activation", "position", "val"]] # Rest of columns constant and thus irrelevant
            grouped_coords = filtered.groupby(by = ["activation"])
            
            if preserve_coord_orgnames : 
                coord_name = str((eval_type, category, reducer))
            else :
                coord_name = f"$x_{coord_dim}$" # Generic dimension number to avoid confusion  

            # Groupby kept giving erroneous KeyErrors that activations were not present despite being so
            for activation, act_data_indices in grouped_coords.groups.items() :
                # Therefore, need to access each group manually at cost of O(n_activations * n_values) time complexity
                data_this_activation = filtered.loc[act_data_indices][["position", "val"]]
                data_this_activation.sort_values(by = ["position"], ascending = True, inplace = True)
                
                # This gets rid of position and val since every entry in the df must be a series for proper multiindex df
                data_this_activation = data_this_activation.set_index("position", drop = True)["val"]
                comparison_dict[(activation, coord_name)] = data_this_activation
        
        # Deliberately have to re-cast as a dataframe or else it says it's of type "Never" (likely bug in pandas)
        comparison_multiindex_df = pd.DataFrame(pd.concat(comparison_dict, axis = 1))
        comparison_multiindex_df.columns = pd.MultiIndex.from_tuples(comparison_multiindex_df.columns, 
                                                                     names = ("activation", "coord"))
        
        # This ensures all coordinates for the same activation are grouped together for consistency
        comparison_multiindex_df.sort_index(axis = 1, level = 0, sort_remaining = False, inplace = True) 
        
        return comparison_multiindex_df 
    
    