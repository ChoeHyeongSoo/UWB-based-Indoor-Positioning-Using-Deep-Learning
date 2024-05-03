import numpy as np
import pandas as pd
import math

class AP:
    def __init__(self, x, y, distance):
        self.x = x
        self.y = y
        self.distance = distance

class Trilateration:
    def __init__(self, AP1, AP2, AP3):
        self.AP1 = AP1
        self.AP2 = AP2
        self.AP3 = AP3
    
    def calcUserLocation(self):
        A = 2 * (self.AP2.x - self.AP1.x)
        B = 2 * (self.AP2.y - self.AP1.y)
        C = self.AP1.distance**2 - self.AP2.distance**2 - self.AP1.x**2 + self.AP2.x**2 - self.AP1.y**2 + self.AP2.y**2
        D = 2 * (self.AP3.x - self.AP2.x)
        E = 2 * (self.AP3.y - self.AP2.y)
        F = self.AP2.distance**2 - self.AP3.distance**2 - self.AP2.x**2 + self.AP3.x**2 - self.AP2.y**2 + self.AP3.y**2
        
        user_x = ( (F * B) - (E * C) ) / ( (B * D) - (E * A))
        user_y = ( (F * A) - (D * C) ) / ( (A * E) - (D * B))
        return user_x, user_y
    
df = pd.read_csv('Trilateration/data_tri.csv') 
# df = pd.read_csv('dsets/dsets_lab/data_2d.csv')
# loc = pd.read_csv('dsets/dsets_lab/loc_2d.csv')

distances = df.values[:10]

coor = []
for i in range(len(distances)):
    AP1 = AP(0, 100, distances[i][1])
    AP2 = AP(100, 0, distances[i][2])
    AP3 = AP(100, 100, distances[i][3])
    tri = Trilateration(AP1,AP2,AP3)
    coor.append(tri.calcUserLocation())

print("Check Point")

print(coor)