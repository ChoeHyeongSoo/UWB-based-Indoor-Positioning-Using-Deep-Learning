import re
import serial
import threading
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Device configuration : LAB cuda 위 - 개인 device=cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Serial port configuration
port = "COM4"
baud = 115200
ser = serial.Serial(port, baud, timeout=1)

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

input_size = 4
hidden_size = 128
output_size = 2
num_layers = 2
num_epochs = 100
learning_rate = 0.001

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm_model.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DWM-1000 Module Linux data process ==========================================
def extract_data(line):
    # Define patterns to extract specific data
    pattern1 = r'C4A7\[0.00,6.48,2.02\]=(.*?) '
    pattern2 = r'1374\[4.00,6.48,2.02\]=(.*?) '
    pattern3 = r'116D\[0.00,0.00,2.02\]=(.*?) '
    pattern4 = r'1104\[4.00,0.00,2.02\]=(.*?) '
    pattern5 = r'est(.*?)\r'
    
    matches1 = re.search(pattern1, line)
    matches2 = re.search(pattern2, line)
    matches3 = re.search(pattern3, line)
    matches4 = re.search(pattern4, line)
    matches5 = re.search(pattern5, line)

    if matches1 and matches2 and matches3 and matches4 and matches5:
        data_c4a7 = matches1.group(1).strip()
        data_1374 = matches2.group(1).strip()
        data_116D = matches3.group(1).strip()
        data_1104 = matches4.group(1).strip()
        data_est = matches5.group(1).strip()

        return data_c4a7, data_1374, data_116D, data_1104, data_est
    else:
        return None, None, None, None, None

def readthread(ser, model): 
    seq_length = 10
    data_buffer = []

    while True:
        if ser.in_waiting > 0:
            res = ser.readline().decode('utf-8', errors='ignore')
            c4a7, _1374, _116D, _1104, _est = extract_data(res)
            if c4a7 and _1374 and _116D and _1104:
                try:
                    c4a7 = float(c4a7) / 3e8
                    _1374 = float(_1374) / 3e8
                    _116D = float(_116D) / 3e8
                    _1104 = float(_1104) / 3e8
                    
                    
                    data_buffer.append([_116D, c4a7, _1104, _1374])
                    
                    if len(data_buffer) > seq_length:
                        data_buffer.pop(0)
                    
                    # seq_length 만족하면 테스트 시작
                    if len(data_buffer) == seq_length:
                        input_data = torch.tensor([data_buffer], dtype=torch.float32).to(device)
                        model.eval()
                        with torch.no_grad():
                            output = model(input_data)
                            output = StandardScaler.inverse_transform(output.detach.cpu().numpy())
                            x, y = output[0] # 측정 좌표 plot으로 정확도 확인
                        
                        print(f"Predicted Coordinates: X={x:.2f}, Y={y:.2f}")
                except Exception as e:
                    print(f"Error in processing data: {e}")

def main():
    thread = threading.Thread(target=readthread, args=(ser, model))
    thread.start()
    thread.join()

if __name__ == "__main__":
    main()
    
"""
# 결과 좌표 inverse_transform
# 1) inverse 자체를 안 하고 학습 / 2) inverse 파라미터만 넘겨와서 시리얼 통신

# 입력 (거리) -> ToA 변환하기

# HW 예측좌표 - 모델 예측좌표 비교할 수 있게

Plot으로 비교한 거 보이게 하면 좋음 !
"""