from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from typing import Any
import torch

# CUSTOM
from support.parsing_helpers import singularise
from support.processing_helpers import pad_torch_stack, nan_long

@dataclass
class experimentResult() :
    
    """
    Simple container class for efficiently representing all categories of result from an experiment. 
        Not intended for any complex calculations, unlike expConfig or expVisual.
    
    Params:
        _results: the dictionary of results, where each key is a category and the value is the tensor of results.
        NOTE: If passing in a list of experimentResults, will apply torch.stack and concatenate on the last dimension.
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
class activationResults() :
    
    """Results dataclass for a complete_activation_test(), or related function. 

    Params:
        results: dictionary of triples (eval_type, category, measure_type) which uniquely define a figure key, with the 
        corresponding DataFrame as its value object.
        
        df: implicit attribute calculated during instantiation from results. Represents all data using a 7-coordinate system:
            1. The figure key identifiers (eval_type, category, measure_type)
            2. The axes identifier (reducer)
            3. The specific plot identifier (activation and kf_reducer)
            4. Datapoint ID (position)
        
        All 4 coordinates combined represent exactly 1 datapoint in a given axes object belonging to a figure.

    """
    
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
        
        """Given all 6 possible coordinate types (excluding "position"), filter the results dataframe for all data that 
        satisfies the criteria and output as a new DataFrame object. Not to be confused with results.df.query().
        
        Leaving any coordinate type as None will return all existing valuations as row elements.
        
        Always outputs a DataFrame, not a Series. Will always be 7 columns, one for each coordinate, even if all columns
        specified. If this behaviour is not desired, consider specific_query().
    
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
        
    def specific_query(self, eval_type : str, category : str, measure_type : str, 
                       reducer : str, kf_reducer : str, activation : str, replace_index : bool = True) -> pd.DataFrame :
    
        """Same as ActivationResults.query(), but returns a single-column Pandas DataFrame. 
        Does not accept NoneType coordinate arguments unlike query(). Unlike query(), will not
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
            raise ValueError(f"NoneType parameter given for specific query {query_requirements}")
        
        query_result = self.query(*query_requirements)

        # May not want to keep "position"
        if replace_index :
            query_result.index = query_result["position"]
            query_result.index.name = singularise(category) # Remove the "s", e.g "epochs" -> "epoch", "params" -> "param"

        # Since all coordinates will be identical, no point in keeping the exact coords
        return query_result[["val"]]

    def compare(self, A : dict[str, str], B : dict[str, str], measure_type : str = "epochs") -> None :
        
        query_A = self.query(**A)
        query_B = self.query(**B)
        
        