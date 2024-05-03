import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# Data Preprocess Function===========================================================

# Coordinates shift =======================================================
def loc_zero_mod(loc_data, domain_mean): # 가장 기본 형태, 필요에 따라 디벨롭
    return loc_data - domain_mean

def loc_zero_demod(loc_data, domain_mean):
    return loc_data + domain_mean

# Classify ToA by Domain
def classify_toa_sign(ToA_data, tag_z):
    Classifed_ToA = []
    for i in range(ToA_data.shape[0]):
        if tag_z[i, 2] < 0:
            Classifed_ToA.append(-ToA_data[i, :])
        else:
            Classifed_ToA.append(ToA_data[i, :])
    return np.array(Classifed_ToA)

#==========================================================================

df = pd.read_csv('dsets/dsets_lab/data_10k.csv')
loc = pd.read_csv('dsets/dsets_lab/loc_10k.csv')

anchor = [(0, 0, 20 ), (0, 70, 20), (40, 0, 20), (40, 70, 20)]

# 앵커 도메인의 중간값 계산
domain_mean = np.mean(anchor, axis=0)

x_data = df.values
y_data = loc.values

# 데이터 스케일링
scaler = MinMaxScaler()
x_data_distance = x_data * 3*(10**8)
y_data_mod = loc_zero_mod(loc.values, domain_mean)
x_classified = classify_toa_sign(x_data_distance, y_data_mod)
x_scaled = scaler.fit_transform(x_classified)
y_scaled = scaler.fit_transform(y_data_mod)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
