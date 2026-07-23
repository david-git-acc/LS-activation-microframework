from abc import ABC, abstractmethod
from torch import nn
import torch
from typing import Callable
import numpy as np
from math import ceil, floor

class ActivationNetwork(ABC, nn.Module) :
    
    """Main network class for the project. Inherits from nn.Module and contains methods for recording data and helper
    functions. All neural networks developed must inherit from here or else the program will not work due to lack of methods.
    
    The primary goal of ActivationNetwork is to construct network types where the activation is the only component that varies
    for the given architecture. Hence, it is a key parameter in the instantiation.
    
    """
    
    # Must be class attr so we can access it without having to instantiate the class
    _name : str = ""
    
    def __init__(self, activation : type[nn.Module], n_inputs : int = 1, n_outputs : int = 1) : 
        super().__init__()  
        
        """
        Instantiate an instance of ActivationNetwork for use in training and testing, presumably with the LS project.
        Note that ActivationNetwork is currently configured for binary and multiclass classification; not 
        
        Params:
            activation: the activation function class to use as this network's nonlinear layers. Must be a class; 
            instances are not allowed to prevent state errors and by-reference copies.

            n_inputs: how many inputs the network will need; typically equal to the number of features in the dataset.
            n_outputs: how many outputs the network will need; typically equal to the number of classes in the dataset.
        """
          
        
        self.n_inputs : int = n_inputs
        self.n_outputs : int = n_outputs
        self.activation : type[nn.Module] = activation
        self._structure : nn.Sequential | None = None
        self.recording = False
        self.hook_structures = [] # Record hooks to track and prevent accumulation over time
        
        self.clear_activation_data()

    @abstractmethod
    def generate_structure(self) -> nn.Sequential :
        """Create the structure of an ActivationNetwork as an nn.Sequential block. Specific to each type of network.

        Returns:
            nn.Sequential: The evaluation network structure itself, as a specific instance. Run model(X) to compute.
        """
        pass

    def clear_activation_data(self) -> None :
        
        """Wipe all activation dictionary data, pertaining to activation outputs and gradients (first-order derivatives.)
        """
        
        self.activation_grads : dict[str, torch.Tensor] = {}
        self.activation_outs : dict[str, torch.Tensor] = {}

    def clear_hooks(self) -> None :
        
        """Wipe all hooks currently attached to the network structure. Does not remove existing activation data.
        After doing this, you will not be able to track activation data until using create_activation_hook() or
        create_all_activation_hooks(), or some equivalent method. 
        """
    
        for hook_struct in self.hook_structures : 
            hook_struct.remove()
        self.hook_structures = []


    def create_activation_hook(self, layer_name: str) -> Callable :
        
        """Create an Activation hook to record activation data. Stores activation outputs, and should be registered with
        torch.register_forward_hook(hook(params)). Hooks will NOT activate unless simultaneously training and 
        in recording mode, since otherwise the computations would be redundant.
        
        Params:
            layer_name: the name of the layer to attach the hook to.
        
        Returns:
            Callable: the hook itself. All hooks are functions on the data passing through self.structure.
        """
        
        def layer_activation_hook(module, inp, out) -> None :
            # Do not record unless training and recording to prevent unnecessary overhead.
            if self.training and self.recording :
                
                # Sometimes tuple, sometimes not - this remedies that
                result : torch.Tensor = out[0] if isinstance(out, tuple) else out
                self.activation_outs[layer_name] = result.detach().cpu() # Detaching ironically faster than keeping on GPU
                
        return layer_activation_hook
    
    
    def create_activation_grad_hook(self, layer_name : str) -> Callable :
        
        """Create a gradient activation hook to record activation first-order derivatives throughout then etwork. Please
        register it with register_full_backward_hook(hook(params)). Like with output activations, must be in training
        and recording mode concurrently for the hook to fire to ensure the data is actually used by trackers from categories.
        
        Params:
            layer_name: name of the layer to attach the hook to.
            
        Returns:
            Callable: the gradient hook itself, to be registered and the handle stored in self.hook_structures.
        """
        
        def layer_activation_grad_hook(module, grad_in, grad_out) -> None :
            # Denied from using type hints due to unreasonable type structure - torch hooks really are a mess
            if self.training and self.recording :
                
                result : torch.Tensor = grad_out[0] if isinstance(grad_out, tuple) else grad_out
                self.activation_grads[layer_name] = result.detach().cpu()
                      
        return layer_activation_grad_hook
        
        
    def create_all_activation_hooks(self) -> None :
        
        """Create all activation hooks for all activations across the entire network in a single function. Note
        that this will eliminate all existing activation hooks and data outputs to preserve data sanity.
        """
        
        self.clear_activation_data()
        self.clear_hooks() # Prevent accumulation of hooks
        
        for layer_name, layer in self.activation_layers :

            grad_hook_struct = layer.register_full_backward_hook(self.create_activation_grad_hook(layer_name))
            self.hook_structures.append(grad_hook_struct)
            
            # Track to remove in next call. I don't know why we have to do it like this, it's how torch works
            out_hook_struct = layer.register_forward_hook(self.create_activation_hook(layer_name))
            self.hook_structures.append(out_hook_struct)
        
        
    def record(self, mode : bool = True) -> None :
        """Set to recording mode or non-recording mode.

        Args:
            mode (bool, optional): The mode of recording. Defaults to True.
        """
        self.recording = mode
    
    def layer_widths(self) -> list[int] :
        
        """Extract list of widths of all layers of the network. Only considers layers with an out_features attribute.
        
        Returns:
            list[int]: the list of integers where each entry L[i] indicates the number of neurons for 
            (presumed) linear layer i.
                
        """        
        widths = []
        for _, layer in self.named_modules() :
            if hasattr(layer, "out_features") :
                widths.append(layer.out_features)
        
        return widths
  
    @property
    def activation_layers(self) -> list[tuple[str, nn.Module]] :
        
        """Return the full list of activation function layers complete with module names. 
        
        Effectively a wrapper function around self.named_modules(), inherited from torch.nn.Module.

        Returns:
            _type_: list[tuple[str, nn.Module]]: the list of layers as a 2-tuple containing the strname and layer.
        """
        
        return [(module_name, module) 
                for module_name, module in self.named_modules() 
                if isinstance(module, self.activation)]
  
    @property
    def name(self) -> str :
        
        """String name of the network. Useful for logging.

        Raises:
            ValueError: If network class name not instantiated. Effectively an abstract class attribute.

        Returns:
            str: the name of the network by class. If this fails, try network_class._name instead.
        """
        
        if not self._name : 
            raise ValueError("Name not instantiated for activation network")
        
        return self._name
  
    @property
    def n_activations(self) -> int :
        
        """Get the total number of activation layers in the network. Simple len() wrapper around the
        activation_layers property.

        Returns:
            int: the number of activation layers.
        """
        
        return len( self.activation_layers )
    
    @property
    def width(self) -> int :
        
        """Maximum width of the network. Simple max-wrapper on self.layer_widths().

        Returns:
            int: the number of neurons in the widest linear-like layer (has out_features) of the neural network.
        """
          
        return max(self.layer_widths())
    
    @property
    def length(self) -> int :
        
        """Length of the network in linear-like layers. Simple len wrapper on self.layer_widths().

        Returns:
            int: the number of layers in the network.
        """
        
        return len(self.layer_widths())
    
    @property   
    def structure(self) -> nn.Sequential :
        
        """Get the nn.Sequential structure contained by the network. Abstract class wrapper on _structure.

        Returns:
            nn.Sequential: the network evaluation structure itself.
        """
        
        error_msg = "No sequential structure defined for ActivationNetwork inheriting subclass"
        assert isinstance(self._structure, nn.Sequential), error_msg 
        
        return self._structure
        
    def forward(self, X : torch.Tensor) -> torch.Tensor :
        
        """Run a simple forward pass of the network over (presumably) feature matrix X.

        Returns:
            torch.Tensor: The output predictions of shape (n_samples, n_outputs).
        """
        
        if self.training :
            self.clear_activation_data()
        return self.structure(X)



class ShortNetwork(ActivationNetwork) :
    
    """Short network ActivationNetwork class. The first one I developed, used mainly for training and debugging
    purposes to be able to compute experiments quickly without having to wait too long and check out features.
    
    Not necessarily designed to be a particularly good network, but often has strong performance on small datasets.
    There is absolutely no motivation behind the exact network shape whatsoever. 
    """
    
    _name = "Short"
    
    def __init__(self, activation : type[nn.Module], n_inputs : int = 1, n_outputs : int = 1) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self._structure = self.generate_structure()
        self.create_all_activation_hooks() # Every subclass must do this on instantiation - cannot be done by super

    def generate_structure(self) -> nn.Sequential :
        return nn.Sequential(
            nn.Linear(self.n_inputs, 5),
            self.activation(),
            nn.Linear(5, 10),
            self.activation(),
            nn.Linear(10, 6),
            self.activation(),
            nn.Linear(6, self.n_outputs)
        )

class DiamondNetwork(ActivationNetwork) :
    
    """Diamond ActivationNetwork class, and the main network of choice in the experiment. Called so because the shape
    of the network if visualised (albeit on a log-scale) would appear as a diamond; geometrically increasing numbers 
    of neurons vertically up to a point, then geometrically decreasing, where the start and endpoints are the numbers
    of features present.
    """
    
    _name = "Diamond"
    
    def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1, 
                 full_length : int = 15, max_width : int = 100) :
        super().__init__(activation, n_inputs, n_outputs)
        
        self.full_length = full_length
        self.max_width = max_width
        self._structure = self.generate_structure()
        
        self.create_all_activation_hooks() 
        
    def generate_structure(self) -> nn.Sequential :
        # We add full_length + 1 layers so there are exactly full_length layers
        left_widths = np.geomspace(self.n_inputs, self.max_width, floor(self.full_length / 2) + 1, endpoint = True)
        right_widths = np.geomspace(self.max_width, self.n_outputs, ceil(self.full_length / 2), endpoint = True)
        layer_lengths = left_widths.astype(int).tolist() + right_widths.astype(int).tolist() 
        
        structure = []
        for layer_index, layer_length in enumerate(layer_lengths[1:], start = 1) :
            
            prior_layer_length = layer_lengths[layer_index - 1]
            layer_itself = nn.Linear(prior_layer_length, layer_length)
            
            # Avoid += logic to avoid O(l^2) concat operations
            structure.append(layer_itself)
            structure.append(self.activation()) # The () instantiates a new activation object to avoid sharing of state
            
        # We added an extra activation at the end so get rid of it
        structure.pop()
        
        return nn.Sequential(*structure)

