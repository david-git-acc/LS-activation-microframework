from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from abc import ABC, abstractmethod
import torch
import pandas as pd
import numpy as np 

### CUSTOM
from dataclass_objects import expInput, experimentResult, monitorParams
from support.processing_helpers import params2grad_vector
from support.parsing_helpers import name2index
from support.torch_reducers import arithmetic_mean, donothing_dummy

# HOW TO ADD NEW CATEGORIES:
# To add a new category, define the tracker (experimentTracker), the tester (post_experiment_test_(categoryname)),
# the name and measure types, and put it in the category registry as a categoryParams instance object. Then it will
# be available as a logging option.

# Please add comment lines to demarcate different category types for clarity.

### ABSTRACT DEFINITION CLASSES

@dataclass
class categoryParams() :
    
    """
    Main container for each data recording category.

    Params:
        name: name of the category, e.g "grad".
        
        tracker: the class assigned to track changes in data throughout experiments. Must implement the 
        methods data, track(), and record(). Class name should always be f"{category}ExperimentTracker".
        
        measure_types: tuple containing all different ways to measure over the data. E.g ("epochs", "params").
        These are converted into dimension indices by the name2index helper function.
        
        _tester: the tester function used to perform all test functions on the extracted data, e.g 
        "post_experiment_test_grad". Class name should always be f"post_experiment_test_{category}".
    """
    
    name : str
    tracker : type[categoryExperimentTracker]
    tester : Callable
    measure_types : tuple[str, ...] = ()
    
class categoryExperimentTracker(ABC) :
    
    """
    Abstract superclass for generating experiment trackers for each category (grad, testloss, testpreds+). 
    Designed to avoid bloating experiment() from activation_testing with increasingly more lines of boilerplate
    and create a more readable design and recording pattern. 
    
    Every category requires 3 key components to compute its observations for later analysis and visualisation:
    
        1. Data storage; the actual tensor that will store all observations from the experiment.
        2. Data tracking; the actual algorithm of extracting data from the experiment object itself.
        3. Data recording; taking the computed tracked data and recording it to data storage.
    
    This abstract superclass enforces implementation of all three methods, guaranteeing standardisation.
    Each experiment tracker is made unique by two identifiers which together form a composite key for the data;

    Params:
        xpi: experimentInput object; stores the actual data in the experiment; this can change. 
        category: the string name for the category (e.g "grad" or "testloss"). 
        data: the torch tensor containing all recorded data of the tracker during experimentation.
    
    """
    
    def __init__(self, xpi : expInput) :
        self.xpi = xpi
        self._category = "placeholder"
        self._data = torch.Tensor([])

    @property
    def category(self) -> str :
        if self._category == "placeholder" :
            raise ValueError(f"No category attribute set. Please define a self._category for tracker {self.__class__.__name__}")
        return self._category
    
    @property
    def data(self) -> torch.Tensor :
        
        """The data attribute stores all the data for this experimentTracker on this category. 

        Raises:
            ValueError: No data attribute provided.

        Returns:
            The torch Tensor containing all the recorded data.
        """
        
        if torch.isnan(self._data).all() :
            raise ValueError(f"No data attribute set. Please define a self._data for tracker {self.__class__.__name__}")
        return self._data
    
    @abstractmethod
    def track(self) -> torch.Tensor :
        """The track() method computes from the experiment data the actual desired statistic, e.g gradient matrix.

        Returns:
            The torch tensor of results, e.g the gradient vector/matrix result.
        """
        pass
        
    @abstractmethod
    def record(self, record_index : int = 0) -> None :
        """The record() method extracts self.track() and then determines how to correctly implant it into self.data.

        Args:
            record_index (int, optional): the index at which to record; used in iteration. Defaults to 0.
        """
        pass
    
######################################## GRAD ##################################################
    
# This is for grad, but it generalises well, so you can use this as a base function to define post_experiment_test on
def post_experiment_test_grad(gms : torch.Tensor, over : str = "epochs",
                          test_suite : tuple[Callable, ...] = (), test_columns : list[str] = [], 
                          kf_reducers : tuple[Callable, ...] = (arithmetic_mean,), 
                          kf_reducer_names : list[str] = ["mean"],
                          expected_ndims : int = 2) -> pd.DataFrame :
    
    """
    Perform the function test suite on a designated set of test functions with k-folds, then collapses over the k-folds using
    an aggregation function (typically mean) and returns results as a Pandas dataframe.
    
    Params:
        gms : list of gradient matrices over folds (3D: (epochs, parameters, folds) - although it need not be this shape)
        over: dimension to check over. Can be set to a different axis manually.
        test_suite: list of functions to test on.
        test_columns: names of each test. If no value given, uses the function names.
        kf_reducers: the aggregation functions to collapse a dimension over. Always becomes mean() if number of folds = 1.
        expected_ndims : number of dimensions that the data is originally meant to be in (before folds). Used for validation.
        
    NOTE: This function is the main function for post experiment testing; testloss and testpreds rely on this one. Also,
    remember that the last dimension must always be the kfold dimension, or bugs will occur - silently or not.


    Returns:
        result_df: Pandas dataframe containing results. Each column is a different agg-type test-type combination.
    """
    
    mp = monitorParams(gms, test_suite, test_columns, kf_reducers, kf_reducer_names)
    mp.validate(expected_ndims = expected_ndims)
     
    # The dimensions to marginalise in are always all dimensions except the dimension we care about 
    # + the kfold dimension (last)
    dim = tuple(i for i in range(len(mp.X.size()) - 1 ) if i != name2index(over) ) 
    
    # Store everything we collect here
    test_results = []
    
    # Store the column names in a given format so easier to store
    df_columns = []
    
    # For each test function we compute the result and collapse all other dimensions using it, to get a 2D array (dim, folds)
    for i, test_func in enumerate( mp.reducers ) : 
        result = test_func(mp.X, dim = dim) # Make sure all test suite functions is NaN-aware + can handle any subset of dims
        
        # Then we collapse the fold dimension in different ways; these are important, will be covered later.
        for j, kfold_aggfunc in enumerate( mp.kf_reducers ) :
        
            kfold_dim = len(result.size()) - 1
        
            collapsed_result = kfold_aggfunc(result, dim = kfold_dim )
            data = collapsed_result.view(-1).numpy() # Convert to NumPy so easier to fit as a dataframe
            test_results.append(data)
            
            df_column_name = (mp.reducer_names[i], mp.kf_reducer_names[j])
            df_columns.append(df_column_name)
    
    assert len(set(df_columns)) == len(df_columns), f"Duplicate df columns. Please check testfuncs and kf_reducers"
    test_results = np.asarray(test_results).T # Transpose to turn features into columns
    result_df = pd.DataFrame(test_results, columns = df_columns)
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = over[:-1] # Kill the "s", we view singularly
        
    return result_df    

    
class gradExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for gradient data. Stores epochs and parameters by default.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "grad"
        self._data = torch.full(size = (xpi.n_captures, *self.xpi.nabla_shape), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return params2grad_vector(self.xpi.anet_model.parameters())
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()
        

######################################## TESTLOSS ##################################################

class testlossExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for testloss data. Stores epochs only by default.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "testloss"
        self._data = torch.full(size = (xpi.n_captures,), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return self.xpi.target_loss(self.xpi.anet_model(self.xpi.X_test_tensor), self.xpi.Y_test_tensor).detach().cpu().item()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index] = self.track()
        

def post_experiment_test_testloss(tl : torch.Tensor, over : None = None,
                          test_suite : None = None, test_columns : list[str] = [], 
                          kf_reducers :  tuple[Callable, ...] = (arithmetic_mean,), 
                          kf_reducer_names : list[str] = ["mean"]) -> pd.DataFrame :
    
    """
    Same as post_experiment_test_grad, but for testloss. Identical logic.
    Note that test_suite, over and test_columns are deprecated because only 1 dimension is supported.

    Returns:
        results_df: dataframe of results. Each column is a different agg-type.
    """
    
    return post_experiment_test_grad(tl, "epochs", ( donothing_dummy, ), ["test loss"], 
                                     kf_reducers, kf_reducer_names, expected_ndims = 1)

######################################## TESTPREDS ##################################################


class testpredsExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for test prediction data. Stores epochs and test samples.
    Note that NaN values will be used as padding whenever stratified K-fold differs in size. As long as using
    a NaN-aware metric e.g those found in helpers (arithmetic_mean, variance, log_average), this will not impact the results.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "testpreds"
        self._data = torch.full(size = (xpi.n_captures, len(xpi.Y_test_tensor)), fill_value = torch.nan) 
    
    def track(self) -> torch.Tensor :
        return torch.argmax(self.xpi.anet_model(self.xpi.X_test_tensor), dim = 1).view(-1).detach().cpu()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()  


def post_experiment_test_testpreds(tps : torch.Tensor, over : str = "test_samples",
                          test_suite : tuple[Callable, ...] = (), test_columns : list[str] = [], 
                          kf_reducers : tuple[Callable, ...] = (arithmetic_mean,), 
                          kf_reducer_names : list[str] = ["mean"]) -> pd.DataFrame :
    
    """
    Same as post_experiment_test_grad, but for test predictions (testpreds). Identical logic.
    
    Returns:
        result_df: Pandas dataframe containing results. Each column is a different agg-type test-type combination.
    """
        
    return post_experiment_test_grad(tps, over, test_suite, test_columns, kf_reducers, kf_reducer_names)

########################################### EVAL METRICS ############################################




#################################### LOGGER CLASS ########################################################



@dataclass 
class categoryExperimentLogger() :
    
    """Main logger class over desired set of categories for a given experiment(); can be used for other functions as well.
    Instantiate one of these loggers for each experiment(). Specify desired categories and they will be tracked.
    
    Also implements polymorphic forms of track(), record() and data to avoid boilerplate manual iteration over trackers.

    Params:
        xpi: set of input parameters for the experiment, and the data source for tracking and recording changes.
        
        categories: a single category or tuple of categories that we are interested in recording from.
    """
    
    xpi : expInput
    categories : tuple[str, ...] | str = "all"
    
    def __post_init__(self) :
        
        if self.categories == "all" :
            self.categories = tuple(category_registry.keys()) # This will grab all existing categories in the registry
        elif isinstance(self.categories, str) :
            self.categories = (self.categories, )
            
        self.categories = tuple(self.categories) # Initialise all relevant trackers
        self.trackers = { cat : category_registry[cat].tracker(self.xpi) for cat in self.categories } # meow

    @property
    def data(self) -> dict[str, torch.Tensor] :
        """Polymorphic implementation of data for the entire logger. 

        Returns:
            dict[str, torch.Tensor]: data recorded from every tracker, identified by category name.
        """
        return { tracker.category : tracker.data for tracker in self.trackers.values()}

    def track(self) -> dict[str, torch.Tensor] :
        """Polymorphic implementation of data for the entire logger. Not directly required due to record().

        Returns:
            dict[str, torch.Tensor]: current tracked data from every tracker, identified by category name.
        """
        return { tracker.category : tracker.track() for tracker in self.trackers.values() }

    def record(self, record_index : int = 0) -> None :
        """Polymorphic implementation of tracker record() to avoid manual iteration.

        Args:
            record_index (int, optional): The index at which to record the tracked data. Defaults to 0.
        """
        for tracker in self.trackers.values() :
            tracker.record(record_index)

    @property
    def result(self) -> experimentResult :
        """Shorthand for returning the appropriate experimentResult object after concluding the experiment, 
        to be passed down to other functions and methods.

        Returns:
            experimentResult: class encapsulating all values in its .results dictionary.
        """
        return experimentResult(self.data)


# When adding any new category, please instantiate and specify all parameters here to avoid data redundancy
# Also, keep it in keyword argument format even if not necessary, for clarity
category_registry : dict[str, categoryParams] = {
    "grad" : categoryParams(name = "grad", 
                            measure_types = ("epochs", "params"),
                            tracker = gradExperimentTracker,
                            tester = post_experiment_test_grad
                            ),
    "testloss" : categoryParams(name = "testloss", 
                                measure_types = ("epochs",),
                                tracker = testlossExperimentTracker,
                                tester = post_experiment_test_testloss
                                ),
    "testpreds" : categoryParams(name = "testpreds", 
                                 measure_types = ("epochs", "test_samples"),
                                 tracker = testpredsExperimentTracker,
                                 tester = post_experiment_test_testpreds
                                 )
}

ordinal_measure_types : set[str] = {"epochs", "layers"}

def is_ordinal(measure_type : str) -> bool :
    return measure_type in ordinal_measure_types

