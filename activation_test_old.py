import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from LIPLo import *

df = pd.read_csv("penguins.csv", index_col = 0)

# SELECT MEASURING DEVICES HERE
activations = [nn.ReLU(), LIPLo(), nn.Tanh()]
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"] # Must be numeric
labels = ["sex"] # Must be categorical
additional_loss_metrics = []

# Hyperparameters
epochs = 500
test_size = 0.2
seed = 42
lr = 0.01
sample_rate = 1

# Fix seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

### FUNCTIONS ###

def transform_df(df : pd.DataFrame, zscale_columns : list[str] = [], encode_columns : list[str] = [] ) -> tuple:
    
    z_scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    df[zscale_columns] = z_scaler.fit_transform(df[zscale_columns])
    df[encode_columns] = encoder.fit_transform(pd.DataFrame( df[encode_columns] ))

    return (z_scaler, encoder)

def pd2torch(df : pd.DataFrame, dtype = torch.float32) -> torch.Tensor :
    
    return torch.tensor(df.values,  dtype = dtype)

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

########################### CODE #############################################

df = df[features + labels].dropna(how = "any").reset_index()

# Separate numeric columns for different preprocessing
numeric_columns = df.select_dtypes(include = "number").columns.tolist()
nonnumerics = [col for col in df.columns if col not in set(numeric_columns)]

df_train, df_test = train_test_split(df, test_size = test_size)

df_train_transformed = df_train.copy()
df_test_transformed = df_test.copy()

z_scaler, encoder = transform_df(df_train_transformed, numeric_columns, nonnumerics)

df_test_transformed[numeric_columns] = z_scaler.transform(df_test_transformed[numeric_columns])
df_test_transformed[nonnumerics] = encoder.transform(df_test_transformed[nonnumerics])

X_train = df_train_transformed[features]
Y_train = df_train_transformed[labels]
X_test = df_test_transformed[features]
Y_test = df_test_transformed[labels]

X_train_tensor = pd2torch(X_train)
Y_train_tensor = pd2torch(Y_train, dtype = torch.long)
X_test_tensor = pd2torch(X_test)
Y_test_tensor = pd2torch(Y_test, dtype = torch.long)


# Dimension compatibility for only 2 class classification problem

if Y_train_tensor.shape[1] == 1 :
    Y_train_tensor = Y_train_tensor.squeeze() #torch.cat([Y_train_tensor, 1 - Y_train_tensor], dim = 1)

if Y_test_tensor.shape[1] == 1 :
    Y_test_tensor = Y_test_tensor.squeeze()

score_matrix = -1 * np.ones(shape = (epochs, len(additional_loss_metrics) + 1, len(activations)))


for k in range(len(activations)) :
    
    activation = activations[k]
        
    my_model = NeuralNetwork(activation)
    my_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(my_model.parameters(), lr=lr)

    my_model.train()
    for i in range(epochs) :
        
        optim.zero_grad()
        predictions = my_model(X_train_tensor)
        loss = my_loss(predictions, Y_train_tensor)
        loss.backward()
        optim.step()
        
        if i % sample_rate == 0 :
            with torch.no_grad() :
                test_predictions = my_model(X_test_tensor)
                
                test_loss = my_loss(test_predictions, Y_test_tensor).item()
                
                test_predictions_numpy = test_predictions.detach().numpy()
                outs = np.argmax(test_predictions_numpy, axis = 1) # maximum over columns, so we get a 1D vector of predictions
                
                score_matrix[i, 0, k] = test_loss
                
                for j, metric in enumerate( additional_loss_metrics ) :
                    valuation = metric(outs, Y_test.values)
                    score_matrix[i, 1+j, k] = valuation
                
                results = encoder.inverse_transform(outs.reshape(-1, 1))
            

    my_model.eval()
    with torch.no_grad() :
        
        test_predictions = my_model(X_test_tensor)
        
        test_loss = my_loss(test_predictions, Y_test_tensor).item()
        
        test_predictions_numpy = test_predictions.detach().numpy()
        outs = np.argmax(test_predictions_numpy, axis = 1) # maximum over columns, so we get a 1D vector of predictions
        
        ba = balanced_accuracy_score(outs, Y_test.values)
        acc = accuracy_score(outs, Y_test.values)
        
        results = encoder.inverse_transform(outs.reshape(-1, 1))
        
        print(f"Test loss: {test_loss}")
        print(f"Balanced accuracy score: {ba}")
        print(f"Accuracy score: {acc}")
        
plt.figure(figsize=(10, 5))

# Cheap hack to avoid having to specify manual names
names = [activation.__class__.__name__ for activation in activations]

for k in range(len( activations )) :

    name = names[k]

    plt.plot(score_matrix[:, 0, k], label=f'Test Loss {(name)}')
    
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()