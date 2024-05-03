import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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
    return loc_data - domain_mean # domain_mean: 앵커 4개의 산술평균(중앙값)

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

df = pd.read_csv('data_10k.csv')
loc = pd.read_csv('loc_10k.csv')

anchor = [(0, 0), (0, 100), (100, 0), (100, 100)]

# 앵커 도메인의 중간값 계산
domain_mean = np.mean(anchor, axis=0)
x_data = df.values
y_data = loc.values

# Data Preprocessing ===========================================================

# 데이터 스케일링
scaler = StandardScaler()
y_data_mod = loc_zero_mod(loc.values, domain_mean)
x_scaled = scaler.fit_transform(x_data)
y_scaled = scaler.fit_transform(y_data_mod)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors    
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train[:, :2], dtype=torch.float32)  # z 좌표를 제외한 x, y 좌표만 선택
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test[:, :2], dtype=torch.float32)  # z 좌표를 제외한 x, y 좌표만 선택

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

# Define hyperparameters
input_size = x_train.shape[1]
hidden_size = 128 
output_size = y_train_tensor.shape[1]  # 출력 크기도 변경
num_epochs = 700
batch_size = 64
learning_rate = 0.001

# Create DataLoader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

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
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.8f}')

# Plot the loss curve
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# 모델 평가 및 스케일 복원
model.eval()
mae_values = []  # 각 샘플의 MAE 값을 저장할 리스트 정의
mse_values = []  # 각 샘플의 MSE 값을 저장할 리스트 정의
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        # 모델 예측
        outputs = model(inputs)
        prediction_scaled = outputs.numpy()

        # 스케일러로 복원
        prediction_original = scaler.inverse_transform(prediction_scaled)
        actual_original = scaler.inverse_transform(targets.numpy())

        # 중심 이동 복원
        prediction_original_demod = loc_zero_demod(prediction_original, domain_mean)
        actual_original_demod = loc_zero_demod(actual_original, domain_mean)
        
        # MAE 계산 및 저장 (z 좌표는 제외)
        mae = mean_absolute_error(actual_original_demod[:, :2], prediction_original_demod[:, :2])
        mse = mean_squared_error(actual_original_demod[:, :2], prediction_original_demod[:, :2])
        mae_values.append(mae)
        mse_values.append(mse)

        all_predictions.append(prediction_original_demod)
        all_targets.append(actual_original_demod)

# Calculate mean MAE and MSE values
mean_mae = np.mean(mae_values)
mean_mse = np.mean(mse_values)
print("Mean MAE Value:", mean_mae)
print("Mean MSE Value:", mean_mse)