import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28) batch_size = 60000 28 * 28 이미지
print(x_test.shape)     # (10000, 28, 28) batch_size = 10000 28 * 28
print(y_train.shape)    # (60000,)  inputdim = 1
print(y_test.shape)     # (10000,)

print(x_train[0])
print(x_train[0].shape)
print('y_train : ', y_train[0])

# plt.imshow(x_train[8000], 'gray')
# plt.imshow(x_train[1])
# plt.show()


# 데이터 전처리 1. 원핫 인코딩
from keras.utils import np_utils
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

# x_train = x_train / 255

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

# x_train = x_train.reshape(60000, 28, 28, 1).asstype('float32') / 255.
# x_test = x_test.reshape(10000, 28, 28, 1).asstype('float32') / 255.

# cnn모델을 구사아하기 위해서 bach_size 행 열 채널
print(x_train.shape)

# 모델

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Flatten

# model = Sequential()
# model.add(Con))
# model.add(Flatten())
# model.add(Dense(1))

# model.summary()