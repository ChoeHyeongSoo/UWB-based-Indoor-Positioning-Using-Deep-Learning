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

# LSTM 설계 =======================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  ## Kernel Error Solve : AVX ?
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

# 모델 초기화
input_size = 10  # 입력 데이터의 크기
hidden_size = 64  # LSTM의 은닉 상태 크기
num_layers = 2  # LSTM 레이어의 수
output_size = 2  # 출력 크기 (좌표 x, y)

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 모델 컴파일
model.compile(loss='mse', optimizer='adam')
model.summary()
