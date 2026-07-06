from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from helpers import dummy_idfunc
from abc import ABC, abstractmethod
import torch
from torch import nn

### CUSTOM
from dataclass_objects import expInputParams, experimentResult
from helpers import params2grad_vector


@dataclass
class categoryParams() :
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
    
    Every category requires 3 things to compute its observations for later analysis and visualisation:
    
        1. Data storage; the actual tensor that will store all observations from the experiment.
        2. Data tracking; the actual way of extracting data from the experiment object itself.
        3. Data recording; taking the computed tracked data and recording it to data storage.
    
    Therefore, this abstract class implementation enforces all three constraints and guarantees standardisation.
    Each experiment tracker is made unique by two identifiers which together form a composite key for the data;

    Params:
        xpi: experimentInputParams object; stores the actual data in the experiment; this can change. 
        category: the string name for the category (e.g "grad" or "testloss"). 
    
    """
    
    def __init__(self, xpi : expInputParams) :
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
        if torch.numel(self._data) == 0 :
            raise ValueError(f"No data attribute set. Please define a self._data for tracker {self.__class__.__name__}")
        return self._data
    
    @abstractmethod
    def track(self) -> torch.Tensor :
        pass
        
    @abstractmethod
    def record(self, record_index : int = 0) -> None :
        pass
    
    
class gradExperimentTracker(categoryExperimentTracker) :
    
    def __init__(self, xpi : expInputParams) :
        super().__init__(xpi)
        self._category = "grad"
        self._data = torch.full(size = (xpi.n, *self.xpi.nabla_shape), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return params2grad_vector(self.xpi.my_model.parameters()).detach().cpu()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()
        

class testlossExperimentTracker(categoryExperimentTracker) :
    
    def __init__(self, xpi : expInputParams) :
        super().__init__(xpi)
        self._category = "testloss"
        self._data = torch.full(size = (xpi.n,), fill_value = torch.nan)
    
    def track(self) -> torch.Tensor :
        return self.xpi.my_loss(self.xpi.my_model(self.xpi.X_test_tensor), self.xpi.Y_test_tensor).item()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index] = self.track()

class testpredsExperimentTracker(categoryExperimentTracker) :
    
    def __init__(self, xpi : expInputParams) :
        super().__init__(xpi)
        self._category = "testpreds"
        self._data = torch.full(size = (xpi.n, len(xpi.Y_test_tensor)), fill_value = torch.nan) 
    
    def track(self) -> torch.Tensor :
        return torch.argmax(self.xpi.my_model(self.xpi.X_test_tensor), dim = 1).view(-1)     
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()  

@dataclass 
class categoryExperimentLogger() :
    xpi : expInputParams
    categories : tuple[str, ...] | str = "all"
    
    def __post_init__(self) :
        
        if self.categories == "all" :
            self.categories = tuple(category_registry.keys())
        elif isinstance(self.categories, str) :
            self.categories = (self.categories, )
            
        self.categories = tuple(self.categories)
        self.trackers = { cat : category_registry[cat].tracker(self.xpi) for cat in self.categories }

    @property
    def data(self) -> dict[str, torch.Tensor] :
        return { tracker.category : tracker.data for tracker in self.trackers.values()}

    def track(self) -> dict[str, torch.Tensor] :
        return { tracker.category : tracker.track() for tracker in self.trackers.values() }

    def record(self, record_index : int = 0) -> None :
        for tracker in self.trackers.values() :
            tracker.record(record_index)

    @property
    def result(self) -> experimentResult :
        return experimentResult(self.data)



def category2measure_types(category : str) -> tuple[str, ...] :
    
    return category_registry[category].measure_types

def get_trackers_from_categories(xpi : expInputParams, 
                                 categories : list[str] | tuple[str, ...]) -> list[categoryExperimentTracker] :
    
    return [category_registry[cat].tracker(xpi) for cat in categories]

def get_all_trackers(xpi : expInputParams) -> list[categoryExperimentTracker] :
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