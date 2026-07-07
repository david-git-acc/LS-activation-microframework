from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from helpers import dummy_idfunc
from abc import ABC, abstractmethod
import torch
from torch import nn

### CUSTOM
from dataclass_objects import expInput, experimentResult
from helpers import params2grad_vector


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
    measure_types : tuple[str, ...] = ()
    _tester : Callable = dummy_idfunc
    
    @property
    def tester(self) :
        
        # Need lazy evaluation to make this happen without circular import error
        # I hate that this is necessary, but there is no other way to get this to work without fusing .py files
        if self._tester.__name__ == "dummy_idfunc":
            import activation_testing
            self._tester = getattr(activation_testing, "post_experiment_test_" + self.name)
        
        return self._tester
    
    
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
        xpi: experimentInputParams object; stores the actual data in the experiment; this can change. 
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
        

class testlossExperimentTracker(categoryExperimentTracker) :
    
    """categoryExperimentTracker implementation for testloss data. Stores epochs only by default.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "testloss"
        self._data = torch.full(size = (xpi.n_captures,), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return self.xpi.my_loss(self.xpi.anet_model(self.xpi.X_test_tensor), self.xpi.Y_test_tensor).detach().cpu().item()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index] = self.track()

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


# Helper functions - unused but may become useful later
def category2measure_types(category : str) -> tuple[str, ...] :
    
    return category_registry[category].measure_types

def get_trackers_from_categories(xpi : expInput, 
                                 categories : list[str] | tuple[str, ...]) -> list[categoryExperimentTracker] :
    
    return [category_registry[cat].tracker(xpi) for cat in categories]

def get_all_trackers(xpi : expInput) -> list[categoryExperimentTracker] :
    return [category_registry[cat].tracker(xpi) for cat in category_registry]



# When adding any new category, please instantiate and specify all parameters here to avoid data redundancy
# Also, keep it in keyword argument format even if not necessary, for clarity
category_registry : dict[str, categoryParams] = {
    "grad" : categoryParams(name = "grad", 
                            measure_types = ("epochs", "params"),
                            tracker = gradExperimentTracker
                            ),
    "testloss" : categoryParams(name = "testloss", 
                                measure_types = ("epochs",),
                                tracker = testlossExperimentTracker
                                ),
    "testpreds" : categoryParams(name = "testpreds", 
                                 measure_types = ("epochs", "test_samples"),
                                 tracker = testpredsExperimentTracker
                                 )
}

ordinal_measure_types : set[str] = {"epochs", "layers"}

def is_ordinal(measure_type : str) -> bool :
    return measure_type in ordinal_measure_types