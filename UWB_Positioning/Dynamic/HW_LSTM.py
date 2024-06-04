import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, euclidean

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
loc_df = pd.read_csv('UWB_Positioning/HW/data/AP/groundtruth.csv', header=None, names=['x', 'y'])
toa_df = pd.read_csv('UWB_Positioning/HW/data/AP/filtered_distance.csv', header=None, names=['toa1', 'toa2', 'toa3', 'toa4'])

toa_df = toa_df / 3e8

# Ensure numeric data and drop NaN values
loc_df['x'] = pd.to_numeric(loc_df['x'], errors='coerce')
loc_df['y'] = pd.to_numeric(loc_df['y'], errors='coerce')
toa_df = toa_df.apply(pd.to_numeric, errors='coerce')
loc_df = loc_df.dropna().reset_index(drop=True)
toa_df = toa_df.dropna().reset_index(drop=True)

# Split into training and testing data
test_size = 6000
train_toa_df = toa_df[:-test_size]
train_loc_df = loc_df[:-test_size]
test_toa_df = toa_df[-test_size:]
test_loc_df = loc_df[-test_size:]

# Define rectangle points
rectangle_points = np.array([
    [0.9663, 4.62724],  # Point 1
    [0.9663, 1.61536],  # Point 2
    [3.0635, 1.61536],  # Point 3
    [3.0635, 4.62724]   # Point 4
])

# Data scaling
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_data = scaler_x.fit_transform(train_toa_df.values)
y_data = scaler_y.fit_transform(train_loc_df.values)

test_x_data = scaler_x.transform(test_toa_df.values)

# Create sequences
def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # Sequence length
x_seq, y_seq = create_sequences(x_data, y_data, seq_length)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, shuffle=False)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device) 
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define LSTM model with 2 layers, dropout, and TimeDistributed(Dense)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        out, _ = self.lstm1(x, (h_0, c_0))
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])
        return out

input_size = x_train.shape[2]
hidden_size = 128
output_size = y_train.shape[1]
num_layers = 2
num_epochs = 100
learning_rate = 0.001

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train ========================================
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'lstm_model.pth')

# Load the trained model
model.load_state_dict(torch.load('lstm_model.pth'))

# Prepare testing data
test_x_seq, test_y_seq = create_sequences(test_x_data, scaler_y.transform(test_loc_df.values), seq_length)

test_x_tensor = torch.tensor(test_x_seq, dtype=torch.float32).to(device)

# Predict with the model
model.eval()
with torch.no_grad():
    predictions = model(test_x_tensor)
    predictions = predictions.cpu().numpy()
    predictions = scaler_y.inverse_transform(predictions)

# Function to calculate distance from point to a line segment
def point_to_segment_dist(point, seg_start, seg_end):
    if np.array_equal(seg_start, seg_end):
        return euclidean(point, seg_start)
    seg_vec = seg_end - seg_start
    pt_vec = point - seg_start
    seg_len = np.dot(seg_vec, seg_vec)
    t = np.dot(pt_vec, seg_vec) / seg_len
    if t < 0.0:
        return euclidean(point, seg_start)
    elif t > 1.0:
        return euclidean(point, seg_end)
    proj = seg_start + t * seg_vec
    return euclidean(point, proj)

# Calculate MSE based on distances
mse_distances = []
for pred in predictions:
    min_dist = float('inf')
    for i in range(len(rectangle_points)):
        seg_start = rectangle_points[i]
        seg_end = rectangle_points[(i + 1) % len(rectangle_points)]
        dist = point_to_segment_dist(pred, seg_start, seg_end)
        if dist < min_dist:
            min_dist = dist
    mse_distances.append(min_dist**2)

mse = np.mean(mse_distances)
print(f'MSE based on distance to rectangle: {mse:.4f}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(predictions[:, 0], predictions[:, 1], 'r--', label='Predicted')
plt.scatter(rectangle_points[:, 0], rectangle_points[:, 1], c='blue', label='Rectangle Points')
plt.plot(np.append(rectangle_points[:, 0], rectangle_points[0, 0]), 
         np.append(rectangle_points[:, 1], rectangle_points[0, 1]), 'b-', label='Rectangle')

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.title('Predicted Coordinates and Distance MSE to Rectangle')
plt.show()