import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv('dsets/dsets_lab/data_10k.csv')
loc = pd.read_csv('dsets/dsets_lab/loc_10k.csv')

# print(input_df)
# print("===========================================================")
# print(input_loc)
# print("===========================================================")

# # x_train = input_df[:10000]
# # x_test = input_df[10000:]

# # y_train = input_loc[:10000]
# # y_test = input_loc[10000:]

# # print(x_train.shape)
# # print(x_test.shape)
# # print(y_train.shape)
# # print(y_test.shape)
# # print("===========================================================")

########################## Data Distribution Check ##############################################

import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.hist(df.values, bins=30, alpha=0.5, label='Origin Data')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of Scaled Data')
# plt.legend()
# plt.show()

########################## Data Scaling Step ###################################################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def loc_zero_mod(loc_data): # Domain Scaling Shift
    return loc_data - 50

def loc_zero_demod(loc_data):
    return loc_data + 50

# 데이터 스케일링
scaler = MinMaxScaler()
x_data = df.values
y_data = loc.values
# y_mod = loc_zero_mod(y_data)
x_scaled = scaler.fit_transform(x_data)
# y_scaled = scaler.fit_transform(y_mod)
y_scaled = scaler.fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# # 스케일링된 데이터의 분포 시각화
# plt.figure(figsize=(10, 6))
# plt.hist(x_scaled, bins=30, alpha=0.5, label='Scaled Data')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of Scaled Data')
# plt.legend()
# plt.show()

########################## Model Set Level ###################################################

# 하이퍼파라미터 설정
epochs = 100    # epoch, batch, lr 조정 +@) 데이터 스케일링, 정규화 확인 : 스케일링 실패, 정규화 ? DropOut?
batch_size = 64

# 신경망 모델 정의 - Sequential ####################################################################
model = Sequential([
    Dense(1024, activation='linear', input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(512, activation='linear'),
    Dropout(0,2),
    Dense(256, activation='linear'),
    Dropout(0.2),
    Dense(3)  # Activate Function 바꿔보며 출력 확인
])

custom_lr = 0.01  # 원하는 학습률 값으로 설정
optimizer = Adam(learning_rate=custom_lr)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse']) 

# 모델 요약 정보 출력
model.summary()
##################################################################################################

# # 모델 정의 - Anchor_Ver ###############################################################################
# input_toa = Input(shape=(x_train.shape[1],), name='input_toa')  # ToA 정보 입력
# input_anchor = Input(shape=(num_anchors,), name='input_anchor')  # 앵커 위치 정보 입력

# # ToA 정보에 대한 은닉층
# hidden_toa = Dense(64, activation='relu')(input_toa)
# hidden_toa = Dropout(0.2)(hidden_toa)

# # 앵커 위치 정보에 대한 은닉층
# hidden_anchor = Dense(64, activation='relu')(input_anchor)
# hidden_anchor = Dropout(0.2)(hidden_anchor)

# # ToA 정보와 앵커 위치 정보를 결합
# concatenated = Concatenate()([hidden_toa, hidden_anchor])

# # 출력층
# output = Dense(3)(concatenated)  # 출력 뉴런의 개수는 예측할 좌표 수에 맞게 설정

# # 모델 생성
# model = Model(inputs=[input_toa, input_anchor], outputs=output)

# # 모델 컴파일 및 요약 정보 출력
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()

########################### Model Test Level ###################################################

# 모델 학습
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

# 각 epoch에서의 loss와 accuracy 출력
for epoch in range(0, epochs, 10):
    # 모델 평가
    loss, mse = model.evaluate(x_test, y_test)
    if epoch % 10 == 0:
        print("Epoch {}/{} - Test 데이터 평가 결과: Loss: {:.4f}, 평균 절대 오차: {:.4f}".format(epoch+1, epochs, loss, mse))
# # 모델 평가
# loss, mse = model.evaluate(x_test, y_test)
# print("\n최종 평가 결과:")
# print("최종 평균 절대 오차:", mse)

# 예측값 확인
predictions = model.predict(x_test)

print("\nEpoch {} 예측값과 실제값 비교:".format(epoch+1))
for i in range(len(x_test)):
    if i % 10 == 0:
        print("예측값: {}, 실제값: {}".format(predictions[i], y_test[i]))

prediction_origin = scaler.inverse_transform(predictions)
y_test_origin = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test_origin, prediction_origin)
print("\n전체 데이터에 대한 MSE:", mse)

# y_pred_original = scaler.inverse_transform(predictions)
# y_test_original = y_data
# for i in range(10):
#     print("예측값: {}, 실제값: {}".format(y_pred_original[i], y_test_original[i]))

# Linear - Linear : loss 1.009 / mae : 0.8700
# Relu - Linear : loss 1.011 / mae : 0.8705
# Relu - ReLu : loss 1.0141 / mae : 0.8719
# L - L - L :  loss: 1.0095 - mae: 0.8704

# ########################### Model Validation Level ###################################################

# #Validation
# import matplotlib.pyplot as plt

# # 모델 학습
# history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

# 그래프로 학습 과정 시각화
plt.figure(figsize=(12, 6))

# 학습 및 검증 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 학습 및 검증 MSE 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Training and Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

# # 모델 평가
# loss, mae = model.evaluate(x_test, y_test)

# print("\nTest 데이터 평가 결과:")
# print("평균 절대 오차:", mae)