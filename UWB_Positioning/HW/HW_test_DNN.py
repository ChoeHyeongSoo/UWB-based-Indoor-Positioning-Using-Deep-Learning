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

# Data Preprocess Function ===========================================================

# Ground Truth 각 포인트
rectangle_points = np.array([
    [0.9663, 4.62724],  # Point 1
    [0.9663, 1.61536],   # Point 2
    [3.0635, 1.61536],    # Point 3
    [3.0635, 4.62724]    # Point 4
])

# GT 세그먼트 함수
def point_to_line_dist(point, line):
    x0, y0 = point
    (x1, y1), (x2, y2) = line

    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return np.hypot(x0 - x1, y0 - y1), (x1, y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        nearest = (x1, y1)
    elif t > 1:
        nearest = (x2, y2)
    else:
        nearest = (x1 + t * dx, y1 + t * dy)
    
    return np.hypot(x0 - nearest[0], y0 - nearest[1]), nearest

# Prediction이랑 GT 최단거리
def point_nearest_rectangle(point, rectangle_points):
    min_distance = float('inf')
    nearest_point = None
    for i in range(len(rectangle_points)):
        line = (rectangle_points[i], rectangle_points[(i + 1) % len(rectangle_points)])
        distance, nearest = point_to_line_dist(point, line)
        if distance < min_distance:
            min_distance = distance
            nearest_point = nearest
    return min_distance, nearest_point

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
hw_x = hw_df.values[:250,:]
hw_y = hw_loc.values[:250,:2]

tag_anchor = [(0.00,0.00), (0.00,6.48), (4.00,0.00), (4.00,6.48)] # HW Anchor 좌표 수정

test_mean = np.mean(tag_anchor, axis=0)

# Data Preprocessing ===================================================================

# 데이터 스케일링
scaler = MinMaxScaler()

hw_loc_mod = loc_zero_mod(hw_y, test_mean)
hw_x = hw_x / 3e8
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

mse_list = []
for pred in all_predictions:
    min_dist, _ = point_nearest_rectangle(pred[:2], rectangle_points)
    mse_list.append(min_dist ** 2)

mean_mse = np.mean(mse_list)
print("Mean MSE Value:", mean_mse)

all_predictions = np.array(all_predictions)

plt.figure(figsize=(8, 8))
plt.plot(rectangle_points[:, 0], rectangle_points[:, 1], 'r-', label='Rectangle')
plt.scatter(all_predictions[:, 0], all_predictions[:, 1], c='g', label='Predicted Path', s=10)
plt.title('Predicted Path with Rectangle')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()