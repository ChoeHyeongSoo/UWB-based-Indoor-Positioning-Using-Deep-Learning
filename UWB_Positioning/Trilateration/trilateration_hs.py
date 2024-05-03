import numpy as np
import math
import sympy as sp

    # 입력으로 거리를 넣는다
    # 거리 계산 필요 x - 있으면 확인용으로 사용은 가능
    # 거리정보(앵커별 1개) 4개로 원을 그려 원 4개의 교점 찾기
    # 방정식을 구현하여, 이 방정식의 해를 구하면, 좌표 x, y, z 구할 수 있다.
    # 행렬식으로 계산해야 할듯

    # 입력 데이터 : ToA => 빛의 속도 곱해서 거리데이터
    # 12,000 : 

# 1. 각 앵커와 태그 사이의 거리를 구하는 함수
def calculate_distance(anchor, tag):
    distance_table = []
    for i in range(len(anchor)):
        distance_table.append(math.sqrt(float(anchor[i][0] - tag[0])**2 + float(anchor[i][1] - tag[1])**2 +float(anchor[i][2] - tag[2])**2))
    return distance_table

# 좌표 계산
def estimate_coor_plus(anchors, distance):
     
    distance_pow = []
    for i in range(len(d)):
        distance_pow.append(round(d[i]**2))

    f = []
    eq = []
    x, y, z = sp.symbols('x y z')
    
    for i in range(len(anchors)):
        f.append(sp.sqrt((anchors[i][0] - x)**2 + (anchors[i][1] - y)**2 + (anchors[i][2] - z)**2))
        eq.append(sp.Eq(f[i], sp.sqrt(distance_pow[i])))
    
    solution = sp.solve(eq, (x, y, z))
    
    return solution
    
anchors = [[0, 0, 1.0], [2.0, 0, 2.0], [0, 2.0, 2.0], [2.0, 2.0, 2.0]]  

tag = [1,1,1]

d = calculate_distance(anchors, tag)
print('앵커-태그 거리:',d)

# ct = estimate_coor_multi(anchors, d)
# print(ct)

coor = estimate_coor_plus(anchors, d)
print(coor)

