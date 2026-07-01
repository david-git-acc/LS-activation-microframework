from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from activations import *
from networks import *
from helpers import *
from config import *
from activation_testing import *
from dataclass_objects import experimentParams
from visualisation import plot_activation_tests

##################### CODE #####################

df = pd.read_csv("datasets/penguins.csv", index_col = 0)
df = df[features + labels].dropna(how = "any").reset_index(drop=True)

test_suite = (arithmetic_mean, variance, log_average )
aggfuncs = ( arithmetic_mean, variance )

# Fix seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

# Because train columns will always be numeric and test column always nonnumeric, we don't need to manually define both lists
feature_transforms = ( (numeric_columns, StandardScaler()), )
label_transforms = ( (nonnumerics, OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1)), )

df_train, df_test = train_test_split(df, test_size = test_size)

experiment_params = experimentParams(df_train, df_test, labels, ShortNetwork, nn.CrossEntropyLoss(),
                        feature_transforms = feature_transforms,
                        label_transforms = label_transforms,
                        activations = (IPLo(), nn.Tanh()),
                        test_suite = test_suite,
                        kfold_aggfuncs = aggfuncs,
                        max_samples = 50,
                        categories = ("grad", "testloss", "testpreds"))

result_dfs = complete_activation_test(experiment_params, verbose = True)

plot_activation_tests(result_dfs, experiment_params, verbose = False)