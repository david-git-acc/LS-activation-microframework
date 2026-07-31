from __future__ import annotations
from abc import ABC, abstractmethod
import torch

### CUSTOM
from dataclass_objects.input_objects import expInput

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
        self._category : str = "placeholder"
        self._data : torch.Tensor = torch.tensor([])

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
        
        if self._data.numel() == 0 :
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
     

ordinal_measure_types : set[str] = {"epochs", "layers"}

def is_ordinal(measure_type : str) -> bool :
    return measure_type in ordinal_measure_types

def measure_type2dim(name : str) -> int :
    
    """Convert a measure type to its corresponding integer dimension for a torch Tensor.
    Because PyTorch does not have full support for named tensors, we assign each way to measure 
    a tensor (e.g epochs, layers, test_samples) to a dimension index in the tensor, depending on
    the category of data. Therefore, multiple measure types can map to the same dimension, since they
    will operate on different category tensors. 
    
    Params:
        name: the name of the measure type.

    Returns:
        int: dimension index of the measure type within the tensor.
    """
    
    mapping = {
        "epochs" : 0,
        "params" : 1,
        "test_samples" : 1,
        "layers" : 1,
        "neurons" : 2,
    }
    
    return mapping[name]