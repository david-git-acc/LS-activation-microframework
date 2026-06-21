from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from visualisation import plot_activation_data
import matplotlib.pyplot as plt
from activations import *
from networks import *
from helpers import *
from config import *
from activation_testing import *
from dataclass_objects import experimentParams
from dataclasses import asdict

##################### CODE #####################

df = df[features + labels].dropna(how = "any").reset_index(drop=True)

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

# Because train columns will always be numeric and test column always nonnumeric, we don't need to manually define both lists
feature_transform_list = [(numeric_columns, StandardScaler())]
label_transform_list = [(nonnumerics, OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]

df_train, df_test = train_test_split(df, test_size = test_size)

experiment_params = experimentParams(df_train, df_test, labels, ShortNetwork,
                        feature_transform_list = feature_transform_list,
                        label_transform_list = label_transform_list,
                        activations = activations,
                        test_suite = test_suite,
                        kfold_aggfuncs = aggfuncs)

complete_activation_loop(**asdict(experiment_params))