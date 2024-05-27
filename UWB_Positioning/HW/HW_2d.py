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

df = pd.read_csv('UWB_Positioning/HW/data/raw/TOA.csv')
loc = pd.read_csv('UWB_Positioning/HW/data/AP/groundtruth_total.csv')

# hw_df = pd.read_csv('HW/data/new_hw/TOA.csv')
# hw_loc = pd.read_csv('HW/data/new_hw/location.csv')

anchor = [(0.00,0.00), (0.00,6.48), (4.00,0.00), (4.00,6.48)]

# 앵커 도메인의 중간값 계산
domain_mean = np.mean(anchor, axis=0)
x_data = df.values
y_data = loc.values

# hw_x = hw_df.values
# hw_y = hw_loc.values
# hw_y = hw_y[:,:2]

tag_anchor = [(0.00,0.00), (0.00,6.48), (4.00,0.00), (4.00,6.48)]

test_mean = np.mean(tag_anchor, axis=0)

# Data Preprocessing ===========================================================

# 데이터 스케일링
scaler = MinMaxScaler()
x_data = x_data / (3.0*(10**8))
y_data_mod = loc_zero_mod(loc.values, domain_mean)
x_scaled = scaler.fit_transform(x_data)
y_scaled = scaler.fit_transform(y_data_mod)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
   
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# test_tensor Extraction : Trilateration&HW 모두 비교하게 추출 - Matlab으로 이거 사용해
x_test_df = pd.DataFrame(x_test, columns=[f'Feature_{i}' for i in range(x_test.shape[1])])
y_test_df = pd.DataFrame(y_test, columns=['Label_X', 'Label_Y'])

x_test_df.to_csv('UWB_Positioning/HW/data/testsets/x_test_total.csv', index=False)
y_test_df.to_csv('UWB_Positioning/HW/data/testsets/y_test_total.csv', index=False)


# hw_test_x = torch.tensor(hw_x_scaled, dtype=torch.float32)
# hw_test_y = torch.tensor(hw_y_scaled, dtype=torch.float32)

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

# Define hyperparameters ============================================
input_size = x_train.shape[1]
hidden_size = 128 
output_size = y_train_tensor.shape[1]
num_epochs = 1000
batch_size = 64
learning_rate = 0.001

# Create DataLoader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(TensorDataset(hw_test_x, hw_test_y), batch_size=batch_size, shuffle=False)

model = DNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===============================================================================================

# Train & Validate Model ========================================================================
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    train_loss = loss.item()
    train_losses.append(train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

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

# model 데이터따라 다르게 저장하기
torch.save(model.state_dict(), 'UWB_Positioning/HW/hw_total_dnn.pth')

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()