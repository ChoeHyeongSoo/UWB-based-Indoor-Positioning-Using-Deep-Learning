import numpy as np
import math
import sympy as sp

    # 입력으로 거리를 넣는다
    # 거리 계산 필요 x - 있으면 확인용으로 사용은 가능
    # 거리정보(앵커별 1개) 4개로 원을 그려 원 4개의 교점 찾기
    # 방정식을 구현하여, 이 방정식의 해를 구하면, 좌표 x, y, z 구할 수 있다.
    # 행렬식으로 계산해야 할듯

# 1. 각 앵커와 태그 사이의 거리를 구하는 함수
def calculate_distance(anchor, tag):
    distance_table = []
    for i in range(len(anchor)):
        distance_table.append(math.sqrt(float(anchor[i][0] - tag[0])**2 + float(anchor[i][1] - tag[1])**2 +float(anchor[i][2] - tag[2])**2))
    return distance_table


# 2.a 정해진 거리(|d|)로 좌표를 추정하는 함수
def estimate_position(anchors, distance):
    
    num_anchors = len(anchors)
    A = np.zeros((num_anchors, 3))
    b = np.zeros((num_anchors, 1))
    
    # 첫 번째 앵커의 좌표
    anchor1 = np.array(anchors[0])
    
    # 각 앵커와 첫 번째 앵커 사이의 방향 벡터와 거리를 이용하여 좌표를 추정
    for i in range(num_anchors):
        A[i] = 2 * (np.array(anchors[i]) - anchor1)
        b[i] = calculate_distance(anchors[i],[0,0,0])**2 - calculate_distance(anchor1,[0,0,0])**2 - distance**2
    
    A_transpose = np.transpose(A)
    ATA = np.dot(A_transpose, A)
    
    ATb = np.dot(A_transpose, b)
    for i in range(num_anchors):
        A[i] = 2 * (np.array(anchors[i]) - anchor1)
        b[i] = np.linalg.norm(np.array(anchors[i]))**2 - np.linalg.norm(anchor1)**2 - distance**2
    
    # A^T * A 와 A^T * b 계산
    A_transpose = np.transpose(A)
    ATA = np.dot(A_transpose, A)
    ATb = np.dot(A_transpose, b)
    
    try:
        # 최소자승법
        estimated_position = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        # 특이행렬 발생 시 유사 역행렬을 사용
        print('Exception: Matrix')
        estimated_position = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
    
    # test 조건
    if np.abs(calculate_distance(estimated_position, anchors[1]) - distance) < 1e-3:
        return "Pass"
    
    return estimated_position.flatten()

# 2.b 정해진 거리(|d|)로 방정식 풀어 좌표 추정
def estimate_coor(anchors, distances):
    
    f = []
    eq = []
    x, y, z = sp.symbols('x y z')
    
    for i in range(len(anchors)):
        f.append(sp.sqrt((anchors[i][0] - x)**2 + (anchors[i][1] - y)**2 + (anchors[i][2] - z)**2))
        eq.append(sp.Eq(f[i], distances[i]))
    
    solution = sp.solve(eq, (x, y, z))
    
    return solution

# 여러 형태
def estimate_coor_multi(anchors, distances_multi):
    # 변수 설정
    x, y, z = sp.symbols('x y z')
    
    # 결과를 저장할 리스트 초기화
    estimated_coordinates = []
    
    # 모든 거리 데이터에 대해 순회하며 추정된 좌표를 계산
    for distances in distances_multi:
        # 방정식 설정
        f = []
        for i in range(len(anchors)):
            f_i = sp.sqrt((anchors[i][0] - x)**2 + (anchors[i][1] - y)**2 + (anchors[i][2] - z)**2) - distances
            f.append(f_i)
        
        # 방정식 시스템 해결
        solution = sp.solve(f, (x, y, z))
        
        # 결과 추가
        estimated_coordinates.append(solution)
    
    return estimated_coordinates

def estimate_coor_plus(anchors, distance_pow):
     
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

coor = estimate_coor(anchors, d)
print(coor)

# ct = estimate_coor_multi(anchors, d)
# print(ct)

dis = []
for i in range(len(d)):
    dis.append(round(d[i]**2))

print(dis)

coor = estimate_coor_plus(anchors, dis)
print(coor)