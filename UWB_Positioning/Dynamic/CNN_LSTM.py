import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

df = pd.read_csv('dsets/dsets_lab/data_10k.csv')
loc = pd.read_csv('dsets/dsets_lab/loc_10k.csv')

anchor = [(0, 0, 50), (0, 100, 60), (100, 0, 60), (100, 100, 60)]

def loc_zero_mod(loc_data): # Domain Scaling Shift
    return loc_data - 50

def loc_zero_demod(loc_data):
    return loc_data + 50

# 데이터 스케일링
scaler = MinMaxScaler()
x_data = df.values
y_data = loc.values
y_mod = loc_zero_mod(y_data)
x_scaled = scaler.fit_transform(x_data)
y_scaled = scaler.fit_transform(y_mod)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create Model =============================================================================
class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, num_layers, fully_connected, device):
        super(CNN_LSTM, self).__init__()
        self.batch_size = batch_size
        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        ) 
        self.lstm = nn.LSTM(input_size = 504,   # LSTM / Batch Sizes는 CIR data 구조 생각해서 조정
                            hidden_size = 32, 
                            num_layers = num_layers,
                            bias = False,
                            dropout = 0.5,
                            bidirectional = True,
                            batch_first=True)

        self.hidden_state, self.cell_state = self.init_hidden()
        
        self.bn = nn.BatchNorm1d(64)
        self.fc_layer = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc_layer_class = nn.Linear(128, out_channels)


    def init_hidden(self): # h_0, c_0 Reference
        hidden_state = torch.zeros(num_layers*2, self.batch_size, 32).to(device)
        cell_state = torch.zeros(num_layers*2, self.batch_size, 32).to(device)

        return hidden_state, cell_state
        
    def forward(self, x):
        x = self.conv1d_layer(x)
        x, _ = self.lstm(x,(self.hidden_state, self.cell_state))
        x = x[:, -1 :].view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc_layer(x)
        x = self.relu(x)
        x = self.fc_layer_class(x)
        x = self.relu(x)

        return x
# ============================================================================================


# Hyper Parameter ========================================================================
num_layer = 
hidden_size = 
c_n = 

    
