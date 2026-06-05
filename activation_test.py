import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from torch.nn.utils import parameters_to_vector
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import random
from LIPLo import *
from helpers import *
from typing import Any, Callable

df = pd.read_csv("penguins.csv", index_col = 0)

# SELECT MEASURING DEVICES HERE
activations = [nn.ReLU(), LIPLo(), nn.Tanh()]
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"] # Must be numeric
labels = ["sex"] # Must be categorical
additional_loss_metrics = []

# Cheap hack to avoid having to specify manual names
names = [activation.__class__.__name__ for activation in activations]

# Hyperparameters
epochs = 500
test_size = 0.2
seed = 42
lr = 0.001
sample_rate = 1
kfold_k = 10

# Fix seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

### FUNCTIONS ###

def pd2torch(df : pd.DataFrame, dtype = torch.float32) -> torch.Tensor :
    
    return np2torch(df.values, dtype = dtype)

def np2torch(array : np.ndarray, dtype = torch.float32) -> torch.Tensor :
    
    converted = torch.tensor(array, dtype = dtype)
    
    if len(converted.shape) > 1 and converted.shape[1] == 1 :
        converted = converted.squeeze()
    
    return converted

def grad2vector(params) -> torch.Tensor :
    # Params must be iterable - a list or iterator

    grad_vector = parameters_to_vector([p.grad if p.grad is not None 
                                        else torch.zeros_like(p) for p in params])
    
    return grad_vector

def pd_data_transformer(transform_list : list[tuple[list[str], Any]]) -> ColumnTransformer :

    """Admits a list of tuples, where each tuple represents a list of dataframe column names to be transformed 
    by the corresponding scaler. Dataframe columns not specified will remain in the dataframe untouched.
    

    Returns:
        The desired transformer.
    """
    
    transformers = []
    
    for columns, chosen_scaler in transform_list :
        transformers.append((f"{chosen_scaler.__class__.__name__}", chosen_scaler, columns))
    
    return ColumnTransformer(transformers, remainder = "passthrough")

n_classes = max(2, len( df[labels].value_counts()))

class NeuralNetwork(nn.Module) :
    def __init__(self, activation) :
        super().__init__()
        self.structure = nn.Sequential(
            nn.Linear(len(features), 5),
            activation,
            nn.Linear(5, 10),
            activation,
            nn.Linear(10, 5),
            activation,
            nn.Linear(5, n_classes)
        )
        
    def forward(self, X) :
        
        evaluated = self.structure(X)
        return evaluated

def experiment(X_train_tensor : torch.Tensor, X_test_tensor : torch.Tensor, 
               Y_train_tensor : torch.Tensor, Y_test_tensor : torch.Tensor, 
               activation : nn.Module, my_loss = nn.CrossEntropyLoss(),
               epochs : int = epochs, lr : float = lr) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]: 
    
    my_model = NeuralNetwork(activation)
    optim = torch.optim.Adam(my_model.parameters(), lr=lr)

    nabla = grad2vector(my_model.parameters())
    gradient_matrix = -1 * torch.ones(size = (epochs, len(nabla)))
    test_loss = -1 * torch.ones(epochs)
    test_predictions = -1 * np.ones(shape = (epochs, len(Y_test_tensor)))

    for i in range(epochs) :
        my_model.train()
        
        optim.zero_grad()
        predictions = my_model(X_train_tensor)
        loss = my_loss(predictions, Y_train_tensor)
        loss.backward()
        optim.step()
        
        # Recompute nabla after optimisation
        nabla = grad2vector(my_model.parameters())
        
        my_model.eval()
        with torch.no_grad() :
            
            test_predictions_torch = my_model(X_test_tensor)
            test_predictions_numpy = test_predictions_torch.detach().cpu().numpy()
            outs = np.argmax(test_predictions_numpy, axis = 1) # maximum over columns, so we get a 1D vector of predictions
            
            gradient_matrix[i, :] = nabla
            test_loss[i] = my_loss(test_predictions_torch, Y_test_tensor).item()
            test_predictions[i, :] = outs.reshape(-1)   
        
        
    return (gradient_matrix, test_loss, test_predictions)

        
def pd_strat_kfold_crossval(df : pd.DataFrame, label : str, activation : nn.Module, 
                         k : int = 10, epochs : int = epochs, 
                         desired_X_transforms = [], desired_Y_transforms = [] ) :
    
    skf = StratifiedKFold(n_splits = k, random_state = seed, shuffle = True)
    
    gradient_matrices = []
    kfold_loss = torch.ones(size = (k, epochs))

    X_train = df.drop(columns = [label])
    Y_train = df[label]
    
    X_transformer = pd_data_transformer(desired_X_transforms)
    Y_transformer = pd_data_transformer(desired_Y_transforms)
    
    fold_i = 0
    for train_index, test_index in skf.split(X_train, Y_train) :
        
        # Boilerplate
        X_train_kf = np2torch( np.asarray(X_transformer.fit_transform( X_train.iloc[train_index] )) )
        X_test_kf = np2torch( np.asarray(X_transformer.transform( X_train.iloc[test_index] )))
        Y_train_kf = np2torch( np.asarray(Y_transformer.fit_transform( Y_train.iloc[train_index] )))
        Y_test_kf = np2torch( np.asarray(Y_transformer.transform( Y_train.iloc[test_index] )))
           
        gm, tl, _ = experiment(X_train_kf, X_test_kf, Y_train_kf, Y_test_kf, activation, epochs = epochs)
        
        gradient_matrices.append(gm)
        kfold_loss[fold_i, :] = tl
        fold_i += 1
    
    return gradient_matrices, kfold_loss
        

########################### CODE #############################################

df = df[features + labels].dropna(how = "any").reset_index()

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

transform_list_x = [(numeric_columns, StandardScaler())]
transform_list_y = [(nonnumerics, OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]

df_train, df_test = train_test_split(df, test_size = test_size)

df_train_transformed = df_train.copy()
df_test_transformed = df_test.copy()
