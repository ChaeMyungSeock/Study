import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa=[]
    for i in range(len(a) - size + 1 ):
        subset = a[i : (i+size)]
        # print(f"{ i+1 }: { subset } ")
        aaa.append(subset)
        # aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

def split_x1(x_list, size):
    x_list = []
    for i in range(len(a) - size + 1 ): #=> 6개짜리
        xset = a[i : (i+size)]
        x_list.append(xset)
    
    print(type(x_list))
    return np.array(x_list)

dataset = split_x(a, size)
dataset2 = split_x1(a, size)

print("=====================")
print(dataset)

print(dataset.shape)
print("=====================")

print(dataset2)

