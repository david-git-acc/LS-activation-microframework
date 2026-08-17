from dataclass_objects.config_objects import expConfig
from torch import nn

# CUSTOM
from visualisation import plot_activation_tests, plot_time2threshold_tests, plot_df_features
from support.config import config
from support.parsing_helpers import safe_dict2params
from activation_testing.general import complete_activation_test, LS_alpha_sensitivity_test, complete_time2threshold_test

experiment_params = expConfig(**safe_dict2params(config, expConfig))

results = complete_activation_test(experiment_params, verbose = True)

coords_df = results.coords2features([("train", "grad", "mean"), ("test", "agrad", "mean")], measure_type = "epochs",
                                    preserve_coord_orgnames = True)

coords_df2 = results.coords2features([("train", "grad", "mean"), ("test", "testloss", "testloss"), 
                                     ("test", "testpreds", "mean")], measure_type = "epochs", preserve_coord_orgnames = False)

coords_df3 = results.coords2features([("train", "aouts", "mean"), ("test", "ls", "mean")], measure_type = "layers", 
                                     preserve_coord_orgnames = False)

coords_df4 = results.coords2features([("train", "testloss", "testloss"), ("test", "testpreds", "mean")], measure_type = "epochs", 
                                     preserve_coord_orgnames = True)

plot_df_features(coords_df, experiment_params, "scatter")
plot_df_features(coords_df2, experiment_params, "scatter")
plot_df_features(coords_df3, experiment_params, "kde")
plot_df_features(coords_df4, experiment_params, "scatter")

thresh_results = complete_time2threshold_test(results, 50)

plot_activation_tests(results, experiment_params, verbose = False)
plot_time2threshold_tests(thresh_results, experiment_params)