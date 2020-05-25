  
# 선형대수 - 벡터

'''
간단히 말하면 벡터(vector)는 벡터끼리 더하거나 상수(scalar)와 곱해지면 새로운 벡터를 생성하는 개념적인 도구다.
더 자세하게는, 벡터는 어떤 유한한 차원의 공간에 존재하는 점들이다. 대부분의 데이터, 특히 숫자로 표현된 데이터는
벡터로 표현할 수 있다. 수많은 사람들의 키, 몸무게, 나이에 대한 데이터가 주어졌다고 해보자. 그렇다면 주어진 데이터를
(키, 몸무게, 나이)로 구성된 3차원 벡터로 표현할 수 있을 것이다.
벡터를 가장 간단하게 표현하는 방법은 여러 숫자의 리스트로 표현하는 것이다.예를 들어 3차원 벡터는 세 개의 숫자로 
구성된 리스트로 표현할 수 있다. 앞으로 벡터는 float 객체를 갖고 있는 리스트인 Vector라는 타입으로 명시할 것이다.
'''
from typing import List

Vector = List[float]

height_weight_age = [70,        # 인치
                     170,       # 파운드,
                     40]        # 나이
                     
                     
grades            = [95,        # 시험1 점수
                     80,        # 시험2 점수
                     75,        # 시험3 점수
                     62]        # 시험4 점수
                     
'''
앞으로 벡터에 대한 산술 연산(arithmetic)을 하고 싶은 경우가 생길 것이다. 파이썬 리스트는 벡터가 아니기 때문에,
이러한 벡터 연산을 해주는 기본적인 도구가 없다. 그러니 벡터 연산을 할 수 있게 해주는 도구를 직접 만들어야함.
두 개의 벡터를 더한다는 것은 각 벡터상에서 같은 위치에 있는 성분끼리 더하는 것이다. 가령 길이가 같은 v와 w라는
두 벡터를 더한다면 v[0] + w[0](첫 번째 성분), v[1] + w[1](두 번째 성분) 으로 구성됨 (만약, 두 벡터의 길이가 
다르다면 두 벡터를 더할 수 없다.) 
벡터 뎃셈은 zip을 사용해서 두 벡터를 묶은 뒤, 각 성분끼리 더하는 리스트 컴프리헨션을 적용하면 된다.
'''

def add(v: Vector, w: Vector) -> Vector:
    '''각 성분끼리 더한다.'''
    assert len(v) == len(w), "vectors must be the same length"

    return[v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

'''
벡터로 구성된 리스트에서 모든 벡터의 각 성분을 더하고 싶은 경우, 첫 번째 성분은 모든 벡터의 첫 번째 성분을
더한 값, 두 번째 성분은 모든 벡터의 두 번째 성분을 더한 값 등으로 구성된다.
'''

   

def vector_sum(vectors: List[Vector]) -> Vector:
    '''모든 벡터의 각 성분들끼리 더한다.'''
    # vectors가 비어있는지 확인
    assert vecotrs, "no vecotrs provided!"

    # 모든 벡터의 길이가 동일한지 확인
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors ), "different sizes!"

    # i번째 결과값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vecotr[i] for vecotr in vectors)
        for i in range(num_elements)]

assert vector_sum([[1,2], [3,4], [5,6], [7,8]]) == [16, 20]

# 스칼라 곱

def scalar_multiply(c: float, v: Vector) -> Vector:
    '''모든 성분을 c로 곱하기'''
    return [c*v_i for v_i in v]


assert vector_multiply([2, [1,2,3]]) == [2, 4, 6]

# 성분별 평균

def vector_mean(vectors: List[Vector]) -> Vector:
    '''각 성분별 평균을 계산'''
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
assert vector_mean([[1,2], [3,4], [5,6]]) == [3,4]

'''
벡터의 내적은(dot product)은 각 성분별 곱한 값을 더해준 값
'''
def dot(v: Vector, w: Vector) -> float:
    """v_1*w_1 + ... + v_n*w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i*w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32      # 1*4 + 2*5 + 3*6

'''
내적의 개념을 사용하면 각 성분의 제곱 값의 합을 쉽게 구할 수 있음
'''

def sum_of_squares(v : Vector) -> float:
    """v_1*w_1 + ... + v_n*w_n"""
    return dot(v, v)

assert sum_of_squares([1,2,3]) == 14 # 1*1 + 2*2 + 3*3

'''
제곱 값의 합을 이용하면 벡터의 크기를 계산할 수 있다.
'''
import math

def manitude (v: Vector) -> float:
    """벡터 v의 크기를 반환"""
    return math.sqrt(sum_of_squares(v))     # math.sqrt는 제곱을 계산해주는 함수

assert magnitude([3, 4]) == 5

'''
이제 두 벡터 간의 거리를 계산
'''
def squared_distance(v: Vector, w : Vecotr) -> float:
    """(v_1- w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    returun sum_of_sqares(subtract(v,w))

def distance(v: Vector, w : Vector) -> float:
    """벡터 v와 w 간의 거리를 계산"""
    return math.sqrt(squared_distance(v, w))

def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v,w))

