import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.init as init
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

df = pd.read_csv('dsets/z_ver/data_10k.csv')
loc = pd.read_csv('dsets/z_ver/loc_10k.csv')

# input_df = df.values
# input_loc = loc.values

# print(input_df)
# print("===========================================================")
# print(input_loc)
# print("===========================================================")

# x_train = input_df[:8000]
# x_test = input_df[8000:]

# y_train = input_loc[:8000]
# y_test = input_loc[8000:]

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print("===========================================================")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 데이터 스케일링
scaler = StandardScaler()
x_data = df.values
y_data = loc.values
x_scaled = scaler.fit_transform(x_data)
y_scaled = scaler.fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the DNN model ===============================================
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
# Weight Initiallizing ==========================================================
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

# Define hyperparameters ==================================================================
input_size = x_train.shape[1]  # Number of input features
hidden_size1 = 256
hidden_size2 = 256
output_size = y_train.shape[1]  # Number of output dimensions
num_epochs = 100
batch_size = 64
learning_rate = 0.001 # 우선, 아담 그냥 사용

# Create DataLoader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size)

# Initialize the model, loss function, and optimizer
model = DNN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())#, lr=learning_rate)

# 모델 초기화
model.apply(weights_init)

# 학습 및 평가 ================================================================

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

# 모델 평가 및 스케일 복원
model.eval()
mse_values = []
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        
        # 모델 예측
        outputs = model(inputs)
        prediction_scaled = outputs.numpy()

        # 스케일러로 복원
        prediction_original = scaler.inverse_transform(prediction_scaled)
        actual_original = scaler.inverse_transform(targets.numpy())

        # MSE 계산
        mse = mean_squared_error(actual_original, prediction_original)
        mse_values.append(mse)

        print(f"Sample {i+1} - Predicted (Original): {prediction_original[0]}, Actual (Original): {actual_original[0]}, MSE: {mse:.4f}")

# 전체 평균 MAE 출력
average_mse = np.mean(mse_values)
print(f"\nAverage MSE: {average_mse:.4f}")

print("############################################")

model.eval()
with torch.no_grad():
    all_predicted = []
    all_ground = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = scaler.inverse_transform(outputs)
        # prediction = np.array(loc_zero_demod(predicted))
        # ground = np.array(loc_zero_demod(scaler.inverse_transform(targets)))
        prediction = np.array(predicted)
        ground = np.array(scaler.inverse_transform(targets))
        all_predicted.extend(prediction)
        all_ground.extend(ground)

mse = mean_squared_error(all_ground, all_predicted)
mae = mean_absolute_error(all_ground, all_predicted)
print(f'Overall MSE: {mse:.4f}')
print(f'Overall MAE: {mae:.4f}')