from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from networks import ShortNetwork, DiamondNetwork
from dataclass_objects.config_objects import expConfig
from torch import nn
from dataclasses import replace

# CUSTOM
from visualisation import plot_activation_tests
from support.config import df, df_train, df_test, config
from support.parsing_helpers import safe_dict2params
from activation_testing import complete_activation_test, LS_alpha_sensitivity_test

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

# Because train columns will always be numeric and test column always nonnumeric, we don't need to manually define both lists
feature_transforms = ( (numeric_columns, StandardScaler()), )
label_transforms = ( (nonnumerics, OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1)), )

experiment_params = expConfig(
                        df_train, df_test, 
                        loss = nn.CrossEntropyLoss(),
                        feature_transforms = feature_transforms,
                        label_transforms = label_transforms,
                        **safe_dict2params(config, expConfig),
                    )

results = complete_activation_test(experiment_params, verbose = True)
plot_activation_tests(results, experiment_params, verbose = False)