from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable
from abc import ABC, abstractmethod
import torch
import pandas as pd
import numpy as np 

### CUSTOM
from dataclass_objects import expInput, experimentResult, testInput
from support.config import config
from support.processing_helpers import params2grad_vector, dfs_settings2tensors
from support.parsing_helpers import name2index, safe_asdict
from support.torch_reducers import donothing_dummy

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
     
