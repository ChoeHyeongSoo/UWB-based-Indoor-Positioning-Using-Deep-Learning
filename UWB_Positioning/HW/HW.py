import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# def classify_toa_sign(ToA_data, tag_z_coord):
#     for i in range(ToA_data.shape[0]):
#         if tag_z_coord[i, 2] < 0:
#             ToA_data[i, :] = -ToA_data[i, :]
#     return ToA_data

#==========================================================================

df = pd.read_csv('HW/tag.csv')
loc = pd.read_csv('HW/Loc.csv')

anchor = [(-4.00,6.60,1.85), (	0.00,6.60,1.85), (-4.00,1.60,1.85), (0.00,0.00,1.85)]

# 앵커 도메인의 중간값 계산
domain_mean = np.mean(anchor, axis=0)

x_data = df.values
y_data = loc.values

# x_train = input_df[:8000]
# x_test = input_df[8000:]

# y_train = input_loc[:8000]
# y_test = input_loc[8000:]

# Data Preprocessing ===========================================================

# 데이터 스케일링
scaler = MinMaxScaler()
y_data_mod = loc_zero_mod(loc.values, domain_mean)
x_classified = classify_toa_sign(x_data, y_data_mod)
x_scaled = scaler.fit_transform(x_classified)
y_scaled = scaler.fit_transform(y_data_mod)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# x_train = x_scaled[:10000]
# x_test = x_scaled[10000:]

# y_train = y_scaled[:10000]
# y_test = y_scaled[10000:]

# ML : Model & Parameters Define ========================================================

# Define the DNN model ===============================================
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define hyperparameters
input_size = x_train.shape[1]
hidden_size = 64
output_size = y_train.shape[1] 
num_epochs = 250
batch_size = 32
learning_rate = 0.001

# Create DataLoader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size,shuffle=True)

# Initialize the model, loss function, and optimizer
model = DNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===============================================================================================

# Train & Validate Model ========================================================================
train_losses = [] # x, y, z 따로
test_losses = []

# Training the model
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Print training loss
    train_loss = loss.item()
    train_losses.append(train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Evaluate the model on validation data
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}')

# Plot the loss curve
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    all_predicted = []
    all_ground = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = scaler.inverse_transform(outputs)
        prediction = np.array(loc_zero_demod(predicted, domain_mean))
        ground = np.array(loc_zero_demod(scaler.inverse_transform(targets), domain_mean))
        all_predicted.extend(prediction)
        all_ground.extend(ground)

mse = mean_squared_error(all_ground, all_predicted)
mae = mean_absolute_error(all_ground, all_predicted)
print(f'Overall MSE: {mse:.4f}')
print(f'Overall MAE: {mae:.4f}')

def mean_squared_error_coordinates(y_true, y_pred):
    # 리스트를 NumPy 배열로 변환
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 데이터 포인트 수
    n = len(y_true)
    
    # 평균 제곱 오차 계산
    # mse = np.sum(np.linalg.norm(y_true - y_pred, axis=1) ** 2)
    mse = np.sum((y_true - y_pred) ** 2) / n
    
    return mse

mse_formula = mean_squared_error_coordinates(all_ground, all_predicted)
# mse = mean_squared_error(all_ground, all_predicted)
print(f'Overall MSE_formula: {mse_formula:.8f}')