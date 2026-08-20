# LS-microframework; a project for testing activations 
<img width="794" height="450" alt="image" src="https://github.com/user-attachments/assets/d3e5a953-1502-42cb-a0e3-d32938f11ffe" />
I created this project as the main testing program for my interests in LS-activations, a subset of the custom S-family of activations (also defined by me) satisfying specific criteria that in theory, make them optimal across the 7 categories that I've defined in the framework. This framework was constructed to solve the problem of standard research scripts requiring manual reevaluation for each category, measure type (e.g over epochs, layers, neurons) and train/test combination as well as visualisation. Given the sheer number of these cases, such a script would not be practical for my purposes, so I built this expandable program to deal with all of them and also flexibly allow any new combinations I might want to add in the future.

For ease of practicality, the framework currently only supports multiple classification targets with arbitrary features and a finite number of $|C|$ classes. This allows me to use cross-entropy loss and famous classification-specific evaluators like Precision, Recall and F1, which do not have Sklearn or Numpy-native regressor implementations. There is nothing in theory preventing the framework from being expanded to regressors, but it is just not the focus of this framework. 

Note that I won't be sharing details of the S activation family in preparation for my future work on the subject area, but LS will be defined later in this readme.

The primary program flow is simply to convert the config.yaml file inputs into a subfolder within experiments, generate the needed PDF visualisations and indexed data CSVs of all described tests, using the prescribed networks and activations from the user in a reasonable time. Almost any justification for any part of the program code can be reduced to this or a subset of this motivation. The only additional component is the ability to compare results from different categories, evaluation types or aggregators after the completion of an experiment; this was needed to allow greater flexibility for any user and myself to help uncover any latent correlations.

Also note that this framework uses custom logging, and using external libraries for this purpose is neither necessary nor desirable. Logging and tracking code and details can be found in categories/ folder, specifically under base_definitions.py and category_registry.py.

## How to run
Use the following steps. This is a Windows repository, but assuming you have pip and Python 3.13 installed, it should theoretically work on any computer.

1. Download the repository in the file location of your choice.
2. (Optional) Create a virtual environment, e.g "venv" and enter it. If you already have all required modules imported, this is unnecessary.
3. In your preferred environment, run `pip install -r requirements.txt`.
4. (Optional) Having read this README and any documentation you wish, customise master.py to do whatever you want it to do. The current instructions already perform much of the intended functionality, but you can also run sensitivity tests or other feature tests if you wish.
5. (Optional) Modify config.yaml to accept any choice of hyperparameters you wish that fall within the accepted domains. Alternatively, leave it as is.
6. Add all datasets you want to test with to the datasets folder, and then assign them a name and their file location in config.py so the program can access it.
7. Run master.py and wait. Once completed, it will generate a folder with name "exp-(network-name)-(initials of activations used)-(hash signature), with all desired results (as .csv file) and figures (as .pdf files).

## Folder-space overview
The project is divided into several static and dynamic folders/subfolders. 

### activation_testing
This folder stores the code segments that perform core experiment  processing operations on the data the bulk of the actual work of the program (not counting evaluation functions, which strictly run *after* the experiment work is completed). It is divided into:

#### general.py
This program stores the main functions, used in master.py, that return ActivationResults dataclass objects; complete_activation_test, LS_alpha_sensitivity_test, and complete_time2threshold_test being the chief examples. They are general by combining mid-level functions from throughout the program to return high-level result objects as these dataclasses. I deliberately designed these "god" functions to minimise hassle/delay for the user and offer a single "do-it" function that bypassed implementation complexity and simply returned the desired results given the configuration expConfig object formed at the top of master.py. 

#### internal.py
This program stores internal functions that are first-order functional calls by general.py; primarily training, result evaluation, KFold cross validation and seed marginalisation. Most functions here do not rely on dataclasses for inputs and retain full function signatures, primarily because they are not intended to be user-facing and instead acquire inputs from higher level functions that *do* rely on dataclasses. It contains the most important function of the codebase, $\texttt{experiment()}$, which performs full network training, tracking and logging of all user-specified category metrics, porting to the correct device and preserving state. Unlike others, experiment() *does* rely on a dataclass, expInput, so that the parameters can be sanitised and reused to ensure validity and reproducibility of results. This dataclass also contains the functions save_state() and reload_state() to guarantee no program-permanent side effects. 

### categories
This folder stores all code pertaining to the 7 *current* key categories of the framework. Standardised code structure is more important here than perhaps any other part of the codebase, due to the requirement to avoid bloating $\texttt{experiment()}$ with an indefinite, unbounded number of categories which would produce unmaintainable spaghetti code. There are 2 main source files, while there is a separate Python source file for each category. 

Note that we define a *measure type* as a different way to measure the same data; e.g either consider how the data changes over **epochs**, over **layers**, over **neurons** or more. The different way of considering the data is called a *measure type*, and each category's data tensor must have exactly as many dimensions as measure types, plus exactly one more dimension as the last one for the KFold dimension. If it's test data, we model the KFold dimension as a single fold to streamline logic and avoid the need to branch across the *eval(uation) type* (train/test). 

##### base_definitions.py
To represent, store and model the data for each category, every category has 3 key responsibilities that must be programmed as concrete methods in the form of the corresponding categoryExperimentTracker abstract class:

- **data**: the actual data container and receptacle for all results calculated from experiments. Must match the actual tensor shape of experiment results, which is why it must be defined manually for each category. Padded with NaNs in case some categories require non-rectangular shapes and also to show where data was left unpopulated.
- **track()**: the method for a category tracker to actually calculate the results for the given category and the current state of the neural network at this particular epoch. Returns a subtensor of the same shape as $\texttt{data}$, but without the epoch dimension in front.
- **record()**: takes in an integer, $\texttt{record index}$, which specifies which index of the epoch dimension of the data attribute to populate with the tracker's \texttt{track()} method at this particular time. Always computes track() (this method is not used independently), then populates the corresponding region of the tensor using the record_index.

The 7 categories are, in chronological order of development:

- **grad**: stores gradient data. Stores data across epochs and all network parameters (params), making _epochs_ and _params_ the measure types here. Should ideally decrease over time (epochs).
- **testloss**: stores test loss, either over the held-back final fold in train data, or the actual test data for eval_type == test. We only examine testloss over epochs, making it the only measure type. Obviously should decrease.
- **testpreds**: the values of the test predictions themselves. Since this experiment framework is intended for multiple, finite label prediction, these values are expected to be between 0 and $|C| - 1$, since ordinal encoding is always used to map labels. Measured over epochs and individual test samples, with the latter being non-ordinal (no intrinsic order; will use a KDE or histplot for visualisation).
- **metrics**: the actual evaluation metrics such as precision, recall, F1, MCC and more. These are calculated over epochs only and done over all test predictions. The method of data collection here is identical to testpreds and thus operates on the same prediction data, but different code is used in post-experiment calculations.
- **agrad**: The gradients of the activation function outputs. Unlike grad, ignores all other parameters than activation functions and their output gradients. Measured over epochs, layers and individual neurons. Neurons are a non-ordinal measure type. Uses NaN-padding up to max network height when a network has different layer heights. Uses ActivationNetwork's backward PyTorch hooks to record data. Should decrease over epochs.
- **aouts**: the outputs of the activation functions. Unlike testpreds, only stores outputs of activations in the intermediate layers and disregards actual final-layer predictions. Measured over epochs, layers and individual neurons. Also uses NaN-padding in the same manner as agrad. Uses ActivationNetwork's forward PyTorch hooks to record data. Expected to increase with epochs.
- **ls**: the current alpha values of the LS activations at each activation layer. This will make more sense after consulting the section on LS in the section on  activations.py. If the activation being measured is not LS-modified, outputs NaN for all entries; if LS-modified but not learnable, outputs the constant LS value that was assigned as the original LS hyperparameter. If LS-modified with learnable alpha, will actually show meaningful results here. Measured over epochs and layers. 

This code stores the measuretype2dim function, which maps each named measure type to its corresponding tensor dimension. If adding a new measure type, you *must* add an entry for the measure type string to the mapping here, or else the code won't know which dimension of tensor data to use.

Finally, this also stores the set of all ordinal measure types, which are just epochs and layers; these are all measure types that support a natural ordering, hence the name. Any measure type not part of this set is considered non-orderable data. If adding a new measure type, be sure to add it to this set if meant to be ordered.

#### category_registry.py
This program stores the categoryParams dataclass, the categoryExperimentLogger concrete class and the category_registry dict. 

the **categoryParams** dataclass is a simple wrapper that stores as a single object everything a category needs to have its data extracted, recorded and then analysed for the framework. It stores the following attributes:
- **name (str)**: the actual name of the category, e.g "grad", "agrad", "metrics".
- **tracker (type[categoryExperimentTracker])**: the concrete subclass of categoryExperimentTracker associated with this category. The naming convention defined is simply the lowercase name of the category, followed by "ExperimentTracker", e.g a subclass named gradExperimentTracker, or lsExperimentTracker. This tracker will track and record data in its data attribute _during the experiments_ for further analysis later. Note that we want the class, not an already-instantiated object, to avoid any continuation of state in future experiments which could disturb outcomes.
- **tester (Callable[[testInput], pd.DataFrame])**: the designated function that given the testInput dataclass, which stores the experiment configuration and tensor result data from the experiment generated by the category's tracker, tests and analyses the results, converting them into a final results DataFrame for easy access later in the program.
- **measure_types (tuple[str, ...])**: the list of all measure types that the category should record over. This is needed to map string measure type to its corresponding numerical dimension in the data translator using the measure_type2dim dictionary map function. 
- **should_increase (bool)**: Whether or not the category is intended to increase. E.g metrics and outputs should increase, but gradients and testloss should not. This is useful for threshold tests.

Once a tracker, tester, name and measure type are defined for a category, it should be used to instantiate a categoryParams object storing all of these attributes which itself should be stored in the **category_registry** dict, which acts as a central source of truth for all information in the framework pertaining to categories. Once stored in this registry, it can be directly called in the config.yaml and executed to provide outputs, provided the code runs, without any further changes. 

I decided to make each categoryParams an object rather than a concrete class off a hypothetical categoryParams abstract superclass because each categoryParams dataclass is intended only as a passive container of data; it has no specific methods or unique behaviours for a particular experiment that would benefit from using an abstract class, and the testers and trackers behave identically across all experiments. While an abstract class would've allowed for a somewhat more compact format by forcing a formal contract between the attributes rather than a manual binding, my choice of creating a separate Python source file acted as its own compartmentalisation and made such boilerplate unnecessary. 

the **categoryExperimentLogger** class is a wrapper around all the categories the user has selected to monitor for this particular experiment. It is a lightweight class storing a tuple of all categories, and the corresponding trackers for each category, and all data for these are sourced directly from the category registry. Like any categoryExperimentTracker, it relies on the same three core attributes of **data**, **track()** and **record()**, merely acting as a single source of execution to avoid boilerplate and provide an abstract surface for experiment recording. The track() and record() functions in particular are just iterators over each category's own track() and record() methods. The data attribute is a dynamic attribute that simply returns the dictionary mapping each category to its tracker's data.

The only unique component of the logger is its result() dynamic attribute, which takes the data and uses it to instantiate an experimentResult dataclass, which runs its own unique logic outside of this program to format the data properly, and is used as the experiment's outward face for further processing by the rest of the framework pipeline.

#### grad.py
The file that stores the tracker and tester functions for the grad category. Note that other components are short (e.g names, should_increase, measure_types) and can be written directly in the category registry, so I don't bother including them here even if it would technically be the more rigorous thing to do, but on the other hand it also allows the category registry to be directly viewed and modified from 1 source in category_registry.py, so there are tradeoffs on both sides. In general, all category files will be stored in the categories folder and only store the tracker and tester functions for standardisation. 

Also note that post_experiment_test_grad is borrowed heavily by most of the other categories except metrics, so things like post_experiment_test_aouts and such are likely just wrappers around this, because the actual algorithm for testing is very similar. Keep that in mind when making any modifications here.

#### testloss.py 
The file storing the tracker and tester functions for the testloss category. There are no reducers / aggregators because there is only 1 measure type and thus no dimensions to reduce / marginalise over. 

#### testpreds.py
The file storing the tracker and tester functions for the testpreds category; test predictions. Note that the "test" in testpreds can refer to either the actual true test data or the held-out k-th fold of K-fold cross validation used as a surrogate test set.

#### metrics.py
The file storing the tracker and tester functions for the metrics category. Uses the same tracker as testpreds, but I kept the code duplicated in case any future update necessitated. However, the tester function is not a wrapper around post_experiment_test_grad here and is instead its own function, mostly because of the need to extract the true labels which requires two-input functions and advanced data extraction. The experimentResult dataclass contains a dict metadata class which is then passed onto the testInput dataclass, which this function can then access to retrieve the exact true transformed data and labels used in the original experiments. 

#### agrad.py
The file storing the tracker and tester functions for the agrad category; activation gradients, or specifically the gradient/first derivative of every output of every layer of the designed activation function instance following a linear layer. Uses ActivationNetwork hooks to store data. Relies on post_experiment_test_grad for test analysis.

#### aouts.py
The file storing the tracker and tester functions for the aouts category; activation outputs for each neuron and layer. Uses ActivationNetwork hooks to store data.

#### ls.py
The file storing the tracker and tester functions for the LS category. Displays a warning if the activation function is not LS-modified, or in other words applied with to_LS(actfunc, *params). 

### dataclass_objects
This folder stores all dataclasses used for the framework's execution, covering a variety of fields from mere data validation and repeated argument usage to result processing and cleaning. In general, these are convenience objects designed to minimise overhead for the processing code and avoid intermingling of administrative and functional code to maintain separation of concerns for different modules / folders. Dataclass objects are divided into 3 key categories:

#### config_objects.py
These dataclasses store configurations of the entire experiment itself and are extremely critical for preserving conditons between sub-experiments. Only 2 dataclasses exist here, but are substantial in scope:

**expConfig**: The general total experiment parameters object generated through config.yaml which is mapped through config.py, finally being instantiated at the top of master.py. This stores the entire infrastructure of the experiment; the dataframes used (both train and test), labels, the network class, the loss function used, the data transforms, reducers (aggregators) as well as key experiment metadata like batch size, lr and number of folds/seeds. In general, it acts as a single source of truth for the experiment configuration, and if any program is to be run with the microframework, it is likely to require this object. While dedicating so much power into a single object is controversial, it also eliminates the need for sub-objects or priority conflicts between any hypothetical lower classes, reduces memory overhead, but most importantly keeps the code clean. If any data is needed for the experiment *itself*, it's going to be here.

**expVisual** is the dataclass generated by expConfig and stores all visual parameters for data visualisation of results made on behalf of expConfig, and the former retains a method to generate the latter. This was made to automate and streamline the process of data visualisation, especially considering that up to 78 independent figures can be produced if using all 7 categories, activation and threshold tests, making manual visualisation per figure infeasible. It also reduces code bloat on visualisation.py by providing a central repository for style and design choices and using methods to bring those choices as dicts over to the program, rather than having the same attributes messily scattered over the entire file, and enabling easier edits. Unlike the rest of the experiment, the user is not empowered with the ability to modify colour, style and design choices at will within config.yaml, since the focus remains on automating the results of computations in a tidy standardised format.

#### input_objects.py
This section stores input objects, which are dataclasses that are designed to serve as repeated inputs for data processing using similar / shared parameters without the need to repeat large function signatures. Using dataclasses here also provides the benefit of moving sanitisation and validation inside the class rather than inside processing functions for SOC. There are two key dataclasses here, the former being used to pass inputs for the experiments themselves, and the latter being used to pass inputs for the post-experiment analyses:

**expInput**: This stores the input to a single experiment; not the entire experiment, only a single run for a single fold or test seed to train a model and have it perform evaluations with categoryExperimentLogger. It stores the feature and label train/test matrices, the current model (with state preserved with the `save_state()` and `reload_state()` methods), the loss, optimiser, learning rate, epochs, categories and all other relevant metadata for training. This is also the dataclass that the categoryExperimentLogger stores a direct reference to, since many of expInput's features are vital for the category trackers to operate. It also enables device switching to CUDA for large data and determines the optimal device for processing to minimise runtime. 

**testInput**: This dataclass is similarly used for repeated argument inputs and data sanitisation, but solely for _after_ all experiment processing with expInput is completed. It serves as the input thread to all `post_experiment_test_(categoryname)` functions and stores the data itself, all reducers and KFold reducers, measure type for this computation (which dimension of the data to _not_ marginalise with the reducers), and a reference to expConfig. 

#### result_objects.py
This program stores result objects; these store the results of computations. Specifically, the ExperimentResult class stores the results of... experiments, and ActivationResults stores all results of post-experiment-test functions performed on each activation, collated into a single object. 

The sole purpose of **ExperimentResult** is to store the collated results of experiments. It contains a few helper methods, but these were made in anticipation for an arbitrary future use case that was never realised during development, but kept as relics that have no remaining use in the program. The only computation it does, if it can be called that, is if given a list of experimentResults rather than the anticipated dictionary mapping measure types to their Tensors, to match the dictionaries of each individual experimentResult by name to produce one giant dictionary of measure types to Tensors with the data as the final fold dimension. This seems small, but actually offloads significant work away from standard computation and saves hassle.

However, **ActivationResults** is vastly more involved, being the format that the main "god function" activation testers like `complete_activation_test()` and `LS_alpha_sensitivity_test()` output. Initially, all such functions returned dictionaries mapping triples of (eval_type, category, measure_type) to dictionaries, and while this was useful for the primary purpose of visualisation, it did not help with any further required manipulation or querying of the results objects for secondary findings. Therefore, it must construct a DataFrame storing the entire pipeline's results in a single object.  

The possible search coordinates / columns (all strings) are:

1. **eval_type**: whether this is train data (KFold crossvalidation results) or test data (seed-marginalised results).
2. **category**: which category the result data belong to.
3. **measure_type**: the measure type used to extract the result data; the non-marginalised tensor dimension of the original experiment results.
4. **reducer**: what aggregation function (e.g mean, log_average, norm, std) was used to marginalise all other dimensions except the measure type and KFold dimension.
5. **kf_reducer** (kfold reducer): what aggregation function was used to marginalise the KFold dimension specifically.
6. **activation**: The activation the ActivationNetwork was using when the results were calculated.

While this consumes significant memory overhead, the benefits mean that now it's possible to query for any desired combination using the developed `select()` and `select_specific()` methods easily and efficiently rather than dig through the original results dict, saving both development and processing time. The former method also supports leaving a search coordinate as None, which will effectively apply no restrictions on that coordinate. 

Finally, ActivationResults() also allows you to compare different "coordinates", where here the definition means a specific projection of search coordinates to isolate a series object, and create a new DataFrame multiindexed by activation and feature, with the primary goal of this being to compare activations across unrelated features not directly seen in the standard activation tests, a "pick-your-own-comparison" method.

### support
The helper functions for the framework. None of these functions were fundamental to the execution (*or else they wouldn't be here*), just used to remove code bloat, promote the separation of concerns or allow for re-use across contexts.

#### config.py
The code used to translate any nontrivial datatypes or references from config.yaml into their intended meanings. For example, when specifying the activations and neural networks, YAML obviously doesn't support encoding these within its markup. So, users instead write the name of the wanted activation / network as well as any modifications they'd like done to them (_e.g LS, batchnorm, adding skip connections_), and this config.py file uses a set of registries to perform the mappings before passing the remaining data onto expConfig via the program-global config dict. This global structure allows for instant and unambiguous access to all components of the main experiment as defined by the user anywhere within the program by importing support/config.py, saving code, time and bugs.  

#### parsing_helpers.py
Code used to do anything related to parsing in string or list data, or other miscellaneous parsing tasks like filtering a function to pass in required arguments only, or converting a plural form to singular (e.g magpies --> magpie). 

#### plotting_helpers.py
Code used to help with plotting; extracting colours with a specified colourmap, determining plot types, or converting categories to more human-readable names that show up on visualisations. 

#### processing_helpers.py
Code used to perform formatting and transformation tasks for Torch tensors, perform padding or other miscellaneous tasks like linear regression; processing tasks not fundamentally related to the microframework goal in any meaningful way.

#### torch_reducers.py
Full set of reducers, e.g variance, standard deviation, mean, log_average, norm, or last_elem. I considered just adding these to processing_helpers.py, but keeping them separate makes it easy to add new ones in the future for any future user/developer.

### datasets
Stores all datasets for the framework. Each one must have a registry mapping dataset string names to the file links in config.py. Left empty in this repository to avoid bloating the file size, which would prevent me from adding any more commits. 

### experiments
Stores all experiments as folders with unique names for their configurations, e.g: "exp-Skip3-Drop20-BN-Diamond-L_L_R-9a6c9d54c3". Each experiment folder is divided into the csvs and figures folders, and then from there into featuers and threshold_tests. 

### Top-level programs
#### activations.py
this file contains the programs for custom activation functions and modifications, primarily the LS transformation operator. $LS : \mathcal{S} \times (0, 1) \to \mathcal {F}$ is defined by:

$$ LS(f(x), \alpha) = \alpha x + (1-\alpha) \frac{f(x)}{f'(0)} $$

and was originally intended to act upon functions in the function family $\mathcal S$, but in principle it can be extended to any function with nonzero derivative at the origin $f'(0) \not = 0$. The code for LS and the custom activation $\text{IPLo}$ (a renamed form of symlog) are both available including with dynamically adjustable alpha, which a specific category, "ls" is defined for in the categories folder subprogram.

The motivation for LS is to force several desirable properties for optimal activations, ensuring linear-like behaviour around the origin:

$$x \approx 0 \implies f_{LS}(x) \approx x + (1-\alpha) \frac{f(0)}{f'(0)}$$

, assuming the target function is continuous and has a defined derivative. Also, $LS$ is invariant to any scalar multiplier of $f$ $cf(x)$ and introduces bijectivity if the function was already injective. However, the bulk of the theoretical benefits appear _only_ when combined with the restrictions of functional membership within $\mathcal S$. Part of the goal of this microframework is to investigate whether and to what degree these theoretical advantages actually translate into superior activation behaviour in real-world datasets, hence the development of the 7 categories. 

LS conversion of a function can be handled in the program using the function `to_LS` with a specified function class and alpha. Note it requires a function class; not an instantiated object! This was done to prevent copying of state to preserve function integrity, as well as minimise boilerplate if needing to reference the class later without forced instantiation of a dummy. An example could be:

`LSFunc = to_LS(nn.Tanh, alpha = 0.05, learnable = True, dtype = torch.float32)`

Following class generation, it can be called to produce a class instance of the LS-modified function as usual and then run it as any other nn.Module activation.

#### master.py
Primary executor program. Run this to take the configuration of config.yaml and produce a full suite of results as a subfolder in experiments/ folder. You can also run sensitivity tests and any other feature extractions from ActivationResults output dataclasses if you want.

#### networks.py
Code for storing the ActivationNetwork abstract class and its instantiated subclasses, of which I only made from necessity ShortNetwork with a simple stock layout and the more robust DiamondNetwork with configurable parameters for max width and length for arbitrary size. I use the latter for all meaningful runs, and the former for when I need to test something fast without waiting forever for training. Each ActivationNetwork is a standard MLP with the only key distinction being that it only runs exactly 1 class of activation function across all activation layers; never switching. By forcing a single activation type per network, it allows us to maintain the choice of activation as an independent variable and thus directly compare results; the foundation of the entire framework.

This part of the code is particularly crucial because ActivationNetwork defines attributes and methods and forms a foundational basis for adding any modifications like BatchNorm, residual conns, dropout, etc that wouldn't be possible for an arbitrary nn.Sequential. It also allows for activation hooks so you can capture that category data in agrad and aouts, at the cost of not being able to easily use torch.compile() for faster performance (or these hooks would fail).  

#### network_mods.py
Code containing modifications to ActivationNetwork concrete classes (Short, DiamondNetwork). A concrete class in this case is a well-defined non-abstract subclass of ActivationNetwork. To avoid boilerplate network code instances and combinatorial explosion in number of manually defined sub-networks (e.g Batchnorm-Diamond with residual block size of 5 and 38% dropout), all network modifiers are functions that act on the original ActivationNetwork concrete class and return a new ActivationNetwork concrete class. 

This also means that all modifications are made lazily, rather than having them available on cue; this is dealt with in config.py to make it *appear* eager by auto-processing network modifications from the config.yaml inside config.py on import, but this is an illusion; there is no pre-batchnormed or similarly modified network classes anywhere in this codebase until called for class creation somewhere. Technically this is an import side effect, but since it makes no sense to have config.yaml *without* config.py and vice versa, and config.py is simply the translator for config.yaml, I view it as acceptable.

The following network modifications are available, although more can be added arbitrarily in a(ny) future update. It was important to make these return classes and not objects so users wouldn't have to repeat the exact same boilerplate code if they wanted to use more than 1 network and also for more flexibility in the future. Because of the factory pattern described above, these also stack on each other as well, allowing the user to run any combination of these modifications in any preferred order, although changing the order may or may not affect functionality depending on the exact modification type.

**to_batchnorm**: Converts an ActivationNetwork to support Batch normalisation. A new instance of PyTorch's BatchNorm is inserted after every linear layer and contains the standard adjustable affine parameters.

**to_layernorm**: Exactly the same as above, but users LayerNorm instead. Likewise also inserts after every linear layer using PyTorch's LayerNorm.

**to_dropout**: Provides a dropout layer to an ActivationNetwork class with a user-specified percentage, and is inserted after each activation in the network. Always runs inplace=False, because that can damage the autograd and incurs testing debt I don't want to incur just for a small memory bonus, which I viewed as not being worth it.

**to_residual**: Provides residual connections for an ActivationNetwork class. Because PyTorch does not offer its own ResidualConnection class, I made my own which gives Identity if number of input layers = number of outputs, or a regular linear layer otherwise, all in a custom ResidualConnection class wrapper. Accepts a block size parameter and defines each block as all layers between 2 activation functions, regardless of the number of layers this margin may involve. So for example there could be a linear layer, skip connection and dropout layer all sandwiched between activation A on the left and activation B on the right, and that'd count as 1 block. If it's on the far left, it's just everything before the first activation, and vice versa on the far right of the network looking left-to-right. 

#### visualisation.py
The main code for visualising all generated results primarily from the ActivationResults dataclass, accepting parameters as dictionaries to avoid formal dependence, but these dictionaries are highly dependent on their data from generating methods *of* expVisual, which itself is derivative of expConfig, the main dataclass that stores the experiment parameters. So retaining the experiment parameters remains highly important, unless you're going to manually define all visual parameters as a dictionary (not recommended; arbitrarily long according to the number of plots, which itself scales with experiment size). 

The primary plotting function, $\texttt{plot data}.py$, does not require any dataclass and only requires a generic plot_params dict, so this is the one truly extensible part of the program without strings attached as plot_params only requires visual characteristics e.g title, legend labels, linestyles, etc. The remaining methods in this class are directly tied to ActivationResults and expVisual dataclass generating methods, so ensure any changes made to those classes are compatible downstream with these functions. 
E.g "plot_df_features" relies on ActivationResults.coords2features / coord2feature and plot_time2threshold_tests depends on threshold_test. 








