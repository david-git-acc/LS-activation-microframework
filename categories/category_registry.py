from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
### CUSTOM
from dataclass_objects.input_objects import expInput
from dataclass_objects.result_objects import ExperimentResult, ActivationResults
from categories.base_definitions import categoryExperimentTracker

from categories.grad import *
from categories.metrics import *
from categories.testloss import *
from categories.testpreds import *
from categories.agrad import *
from categories.aouts import *
from categories.ls import *

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
    should_increase : bool = True
    

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
    categories : tuple[str, ...] 
    
    def __post_init__(self) :
        
        if self.categories == ("all", ) :
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
    def result(self) -> ExperimentResult :
        """Shorthand for returning the appropriate ExperimentResult object after concluding the experiment, 
        to be passed down to other functions and methods.

        Returns:
            ExperimentResult: class encapsulating all values in its .results dictionary.
        """
        return ExperimentResult(self.data)
    

# When adding any new category, please instantiate and specify all parameters here to avoid data redundancy
# Also, keep it in keyword argument format even if not necessary, for clarity
category_registry : dict[str, categoryParams] = {
    "grad" : categoryParams(name = "grad", 
                            measure_types = ("epochs", "params"),
                            should_increase = False,
                            tracker = gradExperimentTracker,
                            tester = post_experiment_test_grad
                            ),
    "testloss" : categoryParams(name = "testloss", 
                                measure_types = ("epochs",),
                                should_increase = False,
                                tracker = testlossExperimentTracker,
                                tester = post_experiment_test_testloss
                                ),
    "testpreds" : categoryParams(name = "testpreds", 
                                 measure_types = ("epochs", "test_samples"),
                                 should_increase = True,
                                 tracker = testpredsExperimentTracker,
                                 tester = post_experiment_test_testpreds
                                 ),
    "metrics" : categoryParams(name = "metrics",
                              measure_types = ("epochs", ),
                              should_increase = True,
                              tracker = metricsExperimentTracker,
                              tester = post_experiment_test_metrics
                              ),
    "agrad" : categoryParams(name = "agrad",
                             measure_types = ("epochs", "layers", "neurons"), 
                             should_increase = False,
                             tracker = agradExperimentTracker,
                             tester = post_experiment_test_agrad
                             ),
    "aouts" : categoryParams(name = "aouts",
                             measure_types = ("epochs", "layers", "neurons"), 
                             should_increase = True,
                             tracker = aoutsExperimentTracker,
                             tester = post_experiment_test_aouts
                             ),
    "ls" : categoryParams(name = "ls",
                          measure_types = ("epochs", "layers"),
                          should_increase = False,
                          tracker = lsExperimentTracker,
                          tester = post_experiment_test_ls
                          ),
}

