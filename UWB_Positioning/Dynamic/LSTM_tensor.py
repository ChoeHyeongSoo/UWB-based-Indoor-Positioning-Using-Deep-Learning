import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# LSTM ============
model = Sequential([
    LSTM(64, input_shape=(x_train[1:].shape), return_sequences=True),
    LSTM(64),
    Dense(2)  # 좌표 (x, y)
])

# 모델 컴파일
model.compile(loss='mse', optimizer='adam')
model.summary()

epochs = 100

# 모델 훈련
history = model.fit(x_train, y_train, epochs=10, batch_size=32)
for epoch in range(0, epochs, 10):
    # 모델 평가
    loss, mse = model.evaluate(x_test, y_test)
    if epoch % 10 == 0:
        print("Epoch {}/{} - Test 데이터 평가 결과: Loss: {:.4f}, 평균 절대 오차: {:.4f}".format(epoch+1, epochs, loss, mse))
