import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris  = load_iris()

x = iris.data[:, (2,3)] # 꽃잎의 길이와 너비
print(x.shape)
y = (iris.target ==0).astype(np.int) # 부채붓꽃(Iris Setosa)인가?
print(y.shape)

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2,0.5]])

print(x)
print(y)

print(y_pred)

'''
퍼셉트론은 클래스 확률을 제공하지 않으며 고정된 임곗값을 기준으로 예측으로 만듬
'''

'''
다층 퍼셉트론(MLP)의 경우는 xor구현 가능 xor => input1 = nand, intput2 = or => xor = and(input1, intput2)  
즉 입력두개를 받아 올 때 and라는 히든 layer를 필요로 함
'''

# And 구현

def AND(x1, x2):
    w1, w2, theta = 0.5,0.5,0.7 # 가중치,theta값 입력
    tmp = x1*w1 + x2*w2 # 수식
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

#파이썬 코드: NAND 구현
def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7 #가중치, theta 값 입력
    tmp = x1*w1 + x2*w2 #수식
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2 #편향
    tmp = np.sum(w*x) +b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
    