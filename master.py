from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from networks import ShortNetwork, DiamondNetwork
from dataclass_objects.config_objects import expConfig
from torch import nn
from dataclasses import replace

# CUSTOM
from visualisation import plot_activation_tests, plot_time2threshold_test
from support.config import config
from support.parsing_helpers import safe_dict2params
from support.plotting_helpers import df2csv
from activation_testing import complete_activation_test, LS_alpha_sensitivity_test, time2threshold_test

# Separate numeric columns for different preprocessing
numeric_columns = config["df"].select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in config["df"].columns if col not in set(numeric_columns)]

# Because train columns will always be numeric and test column always nonnumeric, we don't need to manually define both lists
feature_transforms = ( (numeric_columns, StandardScaler()), )
label_transforms = ( (nonnumerics, OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1)), )

experiment_params = expConfig(
                        loss = nn.CrossEntropyLoss(),
                        feature_transforms = feature_transforms,
                        label_transforms = label_transforms,
                        **safe_dict2params(config, expConfig),
                    )


results = complete_activation_test(experiment_params, verbose = True)
thresh_train = time2threshold_test(results, "train", "testloss", "test loss", 50, False)
thresh_test = time2threshold_test(results, "train", "testloss", "test loss", 50, False)
thresh_train_grad = time2threshold_test(results, "train", "metrics", "mcc", 50)

plot_activation_tests(results, experiment_params, verbose = False)

xvp = experiment_params.exp_vis_params()
plot_time2threshold_test(thresh_train, xvp, "train", "testloss", "test loss", )
plot_time2threshold_test(thresh_test, xvp, "test", "testloss", "test loss", )
plot_time2threshold_test(thresh_train_grad, xvp, "train", "metrics", "mcc", )