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
    return loc_data - domain_mean

def loc_zero_demod(loc_data, domain_mean):
    return loc_data + domain_mean

df = pd.read_csv('UWB_Positioning/HW/data/AP/filtered_distance.csv')
loc = pd.read_csv('UWB_Positioning/HW/data/AP/groundtruth.csv')

hw_df = pd.read_csv('UWB_Positioning/HW/data/raw/TOA.csv')
hw_loc = pd.read_csv('UWB_Positioning/HW/data/raw/location.csv')

anchor = [(0.00,0.00), (0.00,6.48), (4.00,0.00), (4.00,6.48)]

# 앵커 도메인의 중간값 계산
domain_mean = np.mean(anchor, axis=0)
x_data = df.values
y_data = loc.values

# hw_df = hw_df / (3.0*(10**8))
hw_x = hw_df.values[:244,:]
hw_y = hw_loc.values[:244,:2]

tag_anchor = [(0.00,0.00), (0.00,6.48), (4.00,0.00), (4.00,6.48)] # HW Anchor 좌표 수정

test_mean = np.mean(tag_anchor, axis=0)

# Data Preprocessing ===================================================================

# 데이터 스케일링
scaler = MinMaxScaler()

hw_loc_mod = loc_zero_mod(hw_y, test_mean)
hw_x_scaled = scaler.fit_transform(hw_x)
hw_y_scaled = scaler.fit_transform(hw_loc_mod)

hw_test_x = torch.tensor(hw_x_scaled, dtype=torch.float32)
hw_test_y = torch.tensor(hw_y_scaled, dtype=torch.float32)

# ML : Model & Parameters Define =======================================================

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
input_size = hw_test_x.shape[1]
hidden_size = 128 
output_size = hw_test_y.shape[1]  # 출력 크기도 변경
num_epochs = 1000
batch_size = 64
learning_rate = 0.001

# Create DataLoader
# test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(hw_test_x, hw_test_y), batch_size=batch_size, shuffle=False)

# 모델 불러오기
model = DNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('UWB_Positioning/HW/hw_filtered_dnn.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        prediction_original_demod = loc_zero_demod(prediction_original, test_mean)
        actual_original_demod = loc_zero_demod(actual_original, test_mean)
        
        mae = mean_absolute_error(actual_original_demod, prediction_original_demod)
        mse = mean_squared_error(actual_original_demod, prediction_original_demod)
        mae_values.append(mae)
        mse_values.append(mse)
        for i in range(len(prediction_original_demod)):
            all_predictions.append(prediction_original_demod[i])
            all_targets.append(actual_original_demod[i])

# Calculate mean MAE and MSE values
mean_mae = np.mean(mae_values)
mean_mse = np.mean(mse_values)
print("Mean MAE Value:", mean_mae)
print("Mean MSE Value:", mean_mse)
print("min: ", np.min(mse_values))

# 결과 데이터 - Matlab으로 GT랑 MSE 계산해서 HW랑 성능 비교
result = pd.DataFrame(all_predictions, columns=['x', 'y'])
result.to_csv('UWB_Positioning/HW/data/dnn_pred_coor.csv', index=False)

# result.to_excel('추출된_데이터.xlsx', index=False)
# result.scipy.io.savemat('EEG_data.mat')#, {'struct':result.to_dict("list")})