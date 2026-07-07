from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from networks import ShortNetwork
from dataclass_objects import expConfig
from torch import nn

# CUSTOM
from visualisation import plot_activation_tests
from helpers import df, df_train, df_test, config
from activation_testing import complete_activation_test

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

# Because train columns will always be numeric and test column always nonnumeric, we don't need to manually define both lists
feature_transforms = ( (numeric_columns, StandardScaler()), )
label_transforms = ( (nonnumerics, OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1)), )

experiment_params = expConfig(
                        df_train, df_test, 
                        config["labels"], 
                        ShortNetwork, 
                        nn.CrossEntropyLoss(),
                        feature_transforms = feature_transforms,
                        label_transforms = label_transforms,
                        epochs = config["epochs"],
                        activations = config["activations"],
                        activation_names = config["activation_names"],
                        test_functions = config["test_functions"],
                        test_function_names = config["test_function_names"],
                        kfold_aggfuncs = config["kfold_aggfuncs"],
                        kfold_aggfunc_names = config["kfold_aggfunc_names"],
                        max_samples = config["max_samples"],
                        categories = ("grad", "testloss", "testpreds")
                    )

result_dfs = complete_activation_test(experiment_params, verbose = True)

plot_activation_tests(result_dfs, experiment_params, verbose = False)