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

# Define hyperparameters ================================================================
input_size = x_train.shape[1]  # Number of input features (number of anchors)
hidden_size = 256  # Number of neurons in the hidden layer
output_size = y_train.shape[1]  # Number of output dimensions (3 for x, y, z coordinates)
num_epochs = 100
batch_size = 64
learning_rate = 0.01
# =======================================================================================

# Create DataLoader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = DNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

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
        prediction = np.array(loc_zero_demod(predicted))
        ground = np.array(loc_zero_demod(scaler.inverse_transform(targets)))
        all_predicted.extend(prediction)
        all_ground.extend(ground)

mse = mean_squared_error(all_ground, all_predicted)
mae = mean_absolute_error(all_ground, all_predicted)
print(f'Overall MSE: {mse:.4f}')
print(f'Overall MAE: {mae:.4f}')