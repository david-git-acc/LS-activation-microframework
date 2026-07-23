from torch import nn
import torch
from typing import Callable

### CUSTOM
from networks import ActivationNetwork

def update_structure(anet : type[ActivationNetwork], 
                     update_func : Callable, update_name : str) -> type[ActivationNetwork] :
    
    """Helper function to update the structure of an ActivationNetwork class with a given function layer-by-layer,
    then return the modified class for further use. Can be composed with different updates. Does not modify classes
    in-place. Only changes the _structure and _name; no other attributes.
    
    NOTE: all class modification types in network_mods are lazy evaluators; it is impossible to directly modify an 
    existing class with a function at runtime. Instead, we create a new class that comes with pre-prepared instructions
    on what to do to convert the base class into the desired class upon instantiation. These instructions can be stacked
    on top of each other via composition, as our focus on class rather than instance-based instantiation allows this without
    needing to run a full conversion every time we instantiate an object. Returned class outputs are just convenient models.
    
    NOTE: In all modifications, we extract the current_structure as a new instantiated base-network object so that we can
    cannibalise its layers; because it's a new object, it is not referenced anywhere else and we guarantee no by-reference
    hidden copies. However, this does come at the computational cost of instantiating all irrelevant parts, but this is
    epsilon compared to other time factors, predominantly training. 
    
    Params:
        anet: the activation network class (not an instance) to be updated.
        update_func: the function called at each layer of the existing structure to add new layers.
        update_name: the new name of this network. Note it always acts as a prepender on the original network name, 
        e.g "Short" --> "BN-Short" if we are adding batch normalisation.

    Returns:
        type[ActivationNetwork]: the new activation network class. 
    """
    
    class InsertionNetwork(ActivationNetwork) :
        
        _name = update_name + "-" + anet._name
        
        def __init__(self, activation : type[nn.Module], n_inputs : int = 1, n_outputs : int = 1) :
            super().__init__(activation, n_inputs, n_outputs)
            
            # Instantiate a dummy so we can access the class-specific shared attributes
            self.base = anet 
            self._structure = self.generate_structure()
            self.create_all_activation_hooks() 
            
        def generate_structure(self) -> nn.Sequential :
            
            # Current structure of the existing network class; instantiate to gain access
            current_structure = self.base(self.activation, self.n_inputs, self.n_outputs).structure
            new_structure = insert_layers(current_structure, self, update_func)

            return new_structure
    
    return InsertionNetwork


def insert_layers(current_structure : nn.Sequential, network_instance : ActivationNetwork,
                    update_func : Callable) -> nn.Sequential :
    
    """Helper function to insert data between any pair of consecutive layers in a network structure, given
    an update function and the current network object instance. Used to avoid code repetition for key network
    modifications (e.g BatchNorm, Dropout). Used to assist update_structure() to 
    
    Params:
        current_structure: the Sequential network of the original ActivationNetwork, without modifications.
        network_instance: the current instantiation object of network.
        update_func: the per-layer update function that returns an nn.Module per layer, or None if no changes 
        are to be made to that particular layer.
        
    Raises:
        ValueError: Update function does not return None or an nn.Module instance at any point in layer insertion.

    Returns:
        nn.Sequential: New network sequential structure for the new class.
    """
    
    new_structure = []      
    for layer_index in range(len(current_structure) - 1) :
        
        current_layer = current_structure[layer_index]   
        new_structure.append(current_layer)
        
        # This function allows for up to 1 insertion per pair of consec. layers. Limited but functional                                                               
        new_component = update_func(network_instance, current_structure, layer_index) 
        if new_component is None : # Can be set to return None if we do not want to update this layer
            continue
        elif not isinstance(new_component, nn.Module) : 
            error_msg = f"Structural insertion must return nn.Module or None, not {type(new_component)}"
            raise ValueError(error_msg)
        
        new_structure.append(new_component)
    new_structure.append(current_structure[-1])
    
    return nn.Sequential(*new_structure)


def to_batchnorm(anet : type[ActivationNetwork]) -> type[ActivationNetwork] :
    
    """Converts an ActivationNetwork to support Batch normalisation.
    
    Params:
        anet: ActivationNetwork class (not an instance) to update with BatchNorm.
    
    Returns:
        type[ActivationNetwork]: the updated class. 
    """
    
    def update_func(network : ActivationNetwork, current_structure : nn.Sequential, 
                    layer_index : int) -> nn.Module | None :
        
        current_layer = current_structure[layer_index]
        if isinstance(current_layer, nn.Linear) : # BatchNorm usually only goes before linear layers
            return nn.BatchNorm1d(current_layer.out_features)

        return None            
    
    return update_structure(anet, update_func, "BN")


def to_layernorm(anet : type[ActivationNetwork]) -> type[ActivationNetwork] :
    
    """Converts an ActivationNetwork to support LayerNorm normalisation.
    
    Params:
        anet: ActivationNetwork class (not an instance) to update with LayerNorm.
    
    Returns:
        type[ActivationNetwork]: the updated class. 
    """
    
    # Keep redundant parameters present to retain the same signature for consistency
    def update_func(network : ActivationNetwork, current_structure : nn.Sequential, 
                    layer_index : int) -> nn.Module | None :
        
        current_layer = current_structure[layer_index]
        if isinstance(current_layer, nn.Linear) :
            return nn.LayerNorm(current_layer.out_features)

        return None  
         
    return update_structure(anet, update_func, "LN")

def to_dropout(anet : type[ActivationNetwork], p : float = 0.5) -> type[ActivationNetwork] :
    
    """Convert an ActivationNetwork class to support dropout. 
    
    Params:
        anet: the ActivationNetwork class to be updated. 
        p: probability of dropout. E.g p=0.4 means that 40% of neuron activations will be set to 0 in expectation. 
        Using p=0 is equivalent to identity.
        
    Returns:
        type[ActivationNetwork]: the modified network class with DropOut.
    """
    
    def update_func(network : ActivationNetwork, current_structure : nn.Sequential, 
                    layer_index : int) -> nn.Module | None :
        
        current_layer = current_structure[layer_index]
        if isinstance(current_layer, network.activation) :
            return nn.Dropout(p = p, inplace = False)
        
        return None

    return update_structure(anet, update_func, f"Drop{int(p*100)}")

class ResidualConnection(nn.Module) :
    
    """Residual/skip connection implementation for LS project, inheriting directly from nn.Module. Acts as a layer,
    despite containing its own layers. Order of execution is preserved due to nn.Module.named_modules() using DFS traversal.
    
    Designed to mirror the standard skip connection architecture g(x) = f(x) + x as closely as possible, using Identity
    for when out_features == in_features and a learnt Linear layer otherwise.
    """
    
    def __init__(self, structure : nn.Sequential ) :
        
        """Instantiate a residual/skip connection layer, custom to the LS project but may be used elsewhere.
        
        Params:
            structure: the nn.Sequential structure of the network to encapsulate in skip connections.
            Note that this will "swallow" the structure, preserving its execution and gradients but removing its
            direct presence inside the nn.Sequential, although it will still appear in the correct order 
            within self.named_modules(). 
        """

        super().__init__()
        self.structure = structure
        linear_layers = [layer for layer in structure if isinstance(layer, nn.Linear)]
        
        # Preferably should be identity to remove learnable parameters and preserve the g(x) = f(x) + x relationship
        self.proj = nn.Identity()
        if linear_layers :
            self.in_features = linear_layers[0].in_features
            self.out_features = linear_layers[-1].out_features
            
            # However, not always possible, so use a Linear layer just in case. Ideally network should learn to use it correctly.
            if self.in_features != self.out_features : 
                self.proj = nn.Linear(self.in_features, self.out_features) 

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        return self.structure(X) + self.proj(X) # structure = f, proj = id, = f(x) + x.
    
def to_residual(anet : type[ActivationNetwork], block_size : int = 3) -> type[ActivationNetwork] :
      
    """Convert an existing ActivationNetwork class to support residual / skip connections. 
    
    NOTE: As residual connections require substructures, this will appear to "swallow" the network to make it appear 
    as only a set of ResidualConnections when directly inspecting layers in anet.structure. This is an illusion; all layers 
    will remain and this function does not delete any layers. Execution order is preserved via DFS traversal order. 
    
    It is not possible to retain a reference to the original layers in the original structure, or else this risks data
    duplication and multiple paths, which could corrupt the network forward and backward passes.
    
    Params:
        anet: the original ActivationNetwork class to be given skip connections.
        block_size: how many "units", typically linear-activation pairs, should be included per residual connection. The
        more units, the more data is skipped. Defaults to 3.
        
    The block size refers to the number of units per layer. A "unit" is defined in this function as the entire set of 
    network layers up to and including the next activation. So for example, a Linear layer, followed by a batchnorm layer,
    followed by an activation layer would be a "unit", and the dropout, then linear, then batchnorm and next activation
    would be the unit after that. The end unit will just be everything after the last activation in the network sequence.

    Returns:
        type[ActivationNetwork]: the modified network class, now with residual connections.
    """
    
    # Would not fit with the existing linear-like helper functions, so had to make from scratch
    class ResidualNetwork(ActivationNetwork) :
        
        _name = f"Skip{block_size}-" + anet._name
        
        def __init__(self, activation, n_inputs : int = 1, n_outputs : int = 1) :
            super().__init__(activation, n_inputs, n_outputs)        
            self.base = anet # MUST retain references to base network or else can't generate structure at runtime
            self.block_size = block_size
            self._structure = self.generate_structure()
            self.create_all_activation_hooks() 

        def generate_structure(self) -> nn.Sequential:
            
            current_structure = self.base(self.activation, self.n_inputs, self.n_outputs).structure
            
            new_structure = [] 
            up_to_activation = [] # Stores all the layers UP TO the next activation; this is what defines units
            current_block = [] # This stores all the units in the block being loaded at a given time
            units = [] # Stores all units in order, which we will iterate over to create the "blocks" of units
           
            # Unit creation
            for layer in current_structure :
                up_to_activation.append(layer)
                
                # We stop building the unit once we reach an activation
                if isinstance(layer, self.activation) :
                    units.append(up_to_activation)
                    up_to_activation = []
                      
            units.append(up_to_activation)
            
            for unit in units :
                current_block.append(unit)
                
                # We want exactly as many units in the block as the user-specified block size dictates
                if len(current_block) >= block_size :
                    flat_structure = [layer for unit in current_block for layer in unit]
                    sequence = nn.Sequential(*flat_structure)
                    new_structure.append(ResidualConnection(sequence))
                    current_block = []
    
            new_structure += [layer for unit in current_block for layer in unit]
            
            return nn.Sequential(*new_structure)
    
    return ResidualNetwork