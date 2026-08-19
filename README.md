# LS-microframework; a project for testing activations 
<img width="794" height="450" alt="image" src="https://github.com/user-attachments/assets/d3e5a953-1502-42cb-a0e3-d32938f11ffe" />
I created this project as the main testing program for my interests in LS-activations, a subset of the custom S-family of activations (also defined by me) satisfying specific criteria that in theory, make them optimal across the 7 categories that I've defined in the framework. This framework was constructed to solve the problem of standard research scripts requiring manual reevaluation for each category, measure type (e.g over epochs, layers, neurons) and train/test combination as well as visualisation. Given the sheer number of these cases, such a script would not be practical for my purposes, so I built this expandable program to deal with all of them and also flexibly allow any new combinations I might want to add in the future.

Note that I won't be sharing details of the S activation family in preparation for my future work on the subject area, but LS will be defined later in this readme.

The primary program flow is simply to convert the config.yaml file inputs into a subfolder within experiments, generate the needed PDF visualisations and indexed data CSVs of all described tests, using the prescribed networks and activations from the user in a reasonable time. Almost any justification for any part of the program code can be reduced to this or a subset of this motivation. The only additional component is the ability to compare results from different categories, evaluation types or aggregators after the completion of an experiment; this was needed to allow greater flexibility for any user and myself to help uncover any latent correlations.

Note that this framework uses custom logging, and using external libraries for this purpose is neither necessary nor desirable. Logging and tracking code and details can be found in categories/ folder, specifically under base_definitions.py and category_registry.py.

## Folder-space overview
The project is divided into several static and dynamic folders/subfolders. 

### Top-level folders

#### Top-level programs
##### activations.py
this file contains the programs for custom activation functions and modifications, primarily the LS activation application. $LS : \mathcal{F} \times (0, 1) \to \mathcal {F}$ is defined by:

$$ LS(f(x), \alpha) = \alpha x + (1-\alpha) \frac{f(x)}{f'(0)} $$

and was originally intended to act upon functions in the function family $\mathcal S$, but in principle it can be extended to any function. The code for LS, the custom activation $\text{IPLo}$ (a renamed form of symlog) are both available including with dynamically adjustable alpha, which a specific category, "ls" is defined for in the categories folder subprogram.

##### master.py
Primary executor program. Run this to take the configuration of config.yaml and produce a full suite of results as a subfolder in experiments/ folder. You can also run sensitivity tests and any other feature extractions from ActivationResults output dataclasses if you want.

##### networks.py
Code for storing the ActivationNetwork abstract class and its instantiated subclasses, of which I only made from necessity ShortNetwork with a simple stock layout and the more robust DiamondNetwork with configurable parameters for max width and length for arbitrary size. I use the latter for all meaningful runs, and the former for when I need to test something fast without waiting forever for training. 

This part of the code is particularly crucial because ActivationNetwork defines attributes and methods and forms a foundational basis for adding any modifications like BatchNorm, residual conns, dropout, etc that wouldn't be possible for an arbitrary nn.Sequential. It also allows for activation hooks so you can capture that category data in agrad and aouts, at the cost of not being able to easily use torch.compile() for faster performance (or these hooks would fail).  

##### network_mods.py
Code containing modifications to ActivationNetwork concrete classes (Short, DiamondNetwork). A concrete class in this case is a well-defined non-abstract subclass of ActivationNetwork. To avoid boilerplate network code instances and combinatorial explosion in number of manually defined sub-networks (e.g Batchnorm-Diamond with residual block size of 5 and 38% dropout), all network modifiers are functions that act on the original ActivationNetwork concrete class and return a new ActivationNetwork concrete class. 

This also means that all modifications are made lazily, rather than having them available on cue; this is dealt with in config.py to make it *appear* eager by auto-processing network modifications from the config.yaml inside config.py on import, but this is an illusion; there is no pre-batchnormed or similarly modified network classes anywhere in this codebase until called for class creation somewhere. Technically this is an import side effect, but since it makes no sense to have config.yaml *without* config.py and vice versa, and config.py is simply the translator for config.yaml, I view it as acceptable.

The following network modifications are available, although more can be added arbitrarily in a(ny) future update. It was important to make these return classes and not objects so users wouldn't have to repeat the exact same boilerplate code if they wanted to use more than 1 network and also for more flexibility in the future. Because of the factory pattern described above, these also stack on each other as well, allowing the user to run any combination of these modifications in any preferred order, although changing the order may or may not affect functionality depending on the exact modification type.

**to_batchnorm**: Converts an ActivationNetwork to support Batch normalisation. A new instance of PyTorch's BatchNorm is inserted after every linear layer and contains the standard adjustable affine parameters.

**to_layernorm**: Exactly the same as above, but users LayerNorm instead. Likewise also inserts after every linear layer using PyTorch's LayerNorm.

**to_dropout**: Provides a dropout layer to an ActivationNetwork class with a user-specified percentage, and is inserted after each activation in the network. Always runs inplace=False, because that can damage the autograd and incurs testing debt I don't want to incur just for a small memory bonus, which I viewed as not being worth it.

**to_residual**: Provides residual connections for an ActivationNetwork class. Because PyTorch does not offer its own ResidualConnection class, I made my own which gives Identity if number of input layers = number of outputs, or a regular linear layer otherwise, all in a custom ResidualConnection class wrapper. Accepts a block size parameter and defines each block as all layers between 2 activation functions, regardless of the number of layers this margin may involve. So for example there could be a linear layer, skip connection and dropout layer all sandwiched between activation A on the left and activation B on the right, and that'd count as 1 block. If it's on the far left, it's just everything before the first activation, and vice versa on the far right of the network looking left-to-right. 

##### visualisation.py
The main code for visualising all generated results primarily from the ActivationResults dataclass, accepting parameters as dictionaries to avoid formal dependence, but these dictionaries are highly dependent on their data from generating methods *of* expVisual, which itself is derivative of expConfig, the main dataclass that stores the experiment parameters. So retaining the experiment parameters remains highly important, unless you're going to manually define all visual parameters as a dictionary (not recommended; arbitrarily long according to the number of plots, which itself scales with experiment size). 

The primary plotting function, $\texttt{plot\_data}.py$, does not require any dataclass and only requires a generic plot_params dict, so this is the one truly extensible part of the program without strings attached as plot_params only requires visual characteristics e.g title, legend labels, linestyles, etc. The remaining methods in this class are directly tied to ActivationResults and expVisual dataclass generating methods, so ensure any changes made to those classes are compatible downstream with these functions. 
E.g "plot_df_features" relies on ActivationResults.coords2features / coord2feature and plot_time2threshold_tests depends on threshold_test. 








