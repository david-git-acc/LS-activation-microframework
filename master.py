from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from visualisation import plot_activation_data
import matplotlib.pyplot as plt
from activations import *
from networks import *
from helpers import *
from config import *
from activation_testing import *

##################### CODE #####################

df = df[features + labels].dropna(how = "any").reset_index(drop=True)

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

transform_list_x = [(numeric_columns, StandardScaler())]
transform_list_y = [(nonnumerics, OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]

df_train, df_test = train_test_split(df, test_size = test_size)

complete_activation_loop(df_train, df_test, ShortNetwork, labels,
                               transform_list_x = transform_list_x,
                               transform_list_y = transform_list_y,
                               activations = activations,
                               test_suite = test_suite,
                               kfold_aggfuncs = aggfuncs )