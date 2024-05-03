import numpy as np
import pandas as pd

class AP:
    def __init__(self, x, y, z, distance):
        self.x = x
        self.y = y
        self.z = z
        self.distance = distance

class Trilateration:
    def __init__(self, AP1, AP2, AP3, AP4):
        self.AP1 = AP1
        self.AP2 = AP2
        self.AP3 = AP3
        self.AP4 = AP4
    
    def calcUserLocation(self):
        A = np.array([[2 * (self.AP2.x - self.AP1.x), 2 * (self.AP2.y - self.AP1.y), 2 * (self.AP2.z - self.AP1.z)],
                      [2 * (self.AP3.x - self.AP2.x), 2 * (self.AP3.y - self.AP2.y), 2 * (self.AP3.z - self.AP2.z)],
                      [2 * (self.AP4.x - self.AP3.x), 2 * (self.AP4.y - self.AP3.y), 2 * (self.AP4.z - self.AP3.z)]])
        
        b = np.array([self.AP1.distance**2 - self.AP2.distance**2 - self.AP1.x**2 + self.AP2.x**2 - self.AP1.y**2 + self.AP2.y**2 - self.AP1.z**2 + self.AP2.z**2,
                      self.AP2.distance**2 - self.AP3.distance**2 - self.AP2.x**2 + self.AP3.x**2 - self.AP2.y**2 + self.AP3.y**2 - self.AP2.z**2 + self.AP3.z**2,
                      self.AP3.distance**2 - self.AP4.distance**2 - self.AP3.x**2 + self.AP4.x**2 - self.AP3.y**2 + self.AP4.y**2 - self.AP3.z**2 + self.AP4.z**2])
        
        try:
            # 최소자승법으로 좌표를 추정합니다.
            estimated_position = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 특이행렬 발생 시 유사 역행렬을 사용하여 좌표를 추정합니다.
            estimated_position = np.linalg.lstsq(A, b, rcond=None)[0]
        
        return tuple(estimated_position)

df = pd.read_csv('dsets/dsets_lab/data_10k.csv')
loc = pd.read_csv('dsets/dsets_lab/loc_10k.csv')

distances = df.values

AP_Mat = []
coor = []
for i in range(len(df)):
    AP1 = AP(0, 0, 20, distances[i][0])
    AP2 = AP(40, 0, 20, distances[i][1])
    AP3 = AP(0, 70, 20, distances[i][2])
    AP4 = AP(40, 70, 20, distances[i][3])
    tri = Trilateration(AP1,AP2,AP3,AP4)
    coor.append(tri.calcUserLocation())

print("Check Point")

# float64() 형변환
print(coor)