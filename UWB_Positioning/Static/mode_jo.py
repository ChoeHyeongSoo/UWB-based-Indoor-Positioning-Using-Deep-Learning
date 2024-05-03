import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AnchorDynamicModel:
    def __init__(self, data_file, loc_file):
        self.data_file = data_file
        self.loc_file = loc_file
        self.load_data()

    def load_data(self):
        # 데이터 불러오기
        self.data = pd.read_csv(self.data_file)
        self.actual_locations = pd.read_csv(self.loc_file)

    def preprocess_data(self):
        # 특성과 타겟 분리
        self.X = self.data.values  # 4개의 앵커 TOA 값
        self.y = self.actual_locations.values  # 태그의 x, y, z 좌표

        # 데이터 정규화 (Min-Max 스케일링 사용)
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)  # X_scaled로 정규화된 데이터를 저장

    def train_test_split(self, test_size=3000):
        # 훈련 데이터와 테스트 데이터 분리
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42
        )

    def create_model(self):
        # DNN 모델 생성 (함수형 API 사용)
        input_layer = layers.Input(shape=(self.X_train.shape[1],))
        dense1 = layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(input_layer)
        batch_norm1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(batch_norm1)
        batch_norm2 = layers.BatchNormalization()(dense2)
        dense3 = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(batch_norm2)
        batch_norm3 = layers.BatchNormalization()(dense3)

        # 다중 출력 레이어 (활성화 함수를 변경하여 음수 값을 고려함)
        output_layer = layers.Dense(3, activation='linear')(batch_norm3)  # 3개의 출력 뉴런 (x, y, z 좌표)

        self.model = keras.Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self, learning_rate=0.001):
        # 모델 컴파일
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        # 모델 훈련
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate_model(self):
        # 테스트 데이터로 예측
        self.y_pred = self.model.predict(self.X_test)

        # 평가
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)

        print(f"Mean Squared Error on Test Data: {self.mse}")
        print(f"Mean Absolute Error on Test Data: {self.mae}")

        # DNN 알고리즘에서 예측한 결과값과 실제 위치 비교하여 정확도 측정
        self.accuracy = np.mean(np.linalg.norm(self.y_test - self.y_pred, axis=1) < 0.1)  # 정확도를 예를 들어 0.1로 설정
        print(f"Accuracy: {self.accuracy * 100:.2f}%")

        # 각 데이터의 실제 좌표와 예측 좌표 출력
        for i in range(len(self.y_test)):
            print(f"실제 좌표: {self.y_test[i]}, 예측 좌표: {self.y_pred[i]}")

    def run(self):
        self.preprocess_data()
        self.train_test_split()
        self.create_model()
        self.compile_model()
        self.train_model()
        self.evaluate_model()

# 매트랩에서 생성된 데이터 파일 및 위치 파일
data_file = "dsets/data_10k.csv"
loc_file = "dsets/loc_10k.csv"

# 모델 실행
model = AnchorDynamicModel(data_file, loc_file)
model.run()