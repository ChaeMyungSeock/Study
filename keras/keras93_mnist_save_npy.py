# keras91 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import  np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28) batch_size = 60000 28 * 28 이미지
print(x_test.shape)     # (10000, 28, 28) batch_size = 10000 28 * 28
print(y_train.shape)    # (60000,)  inputdim = 1
print(y_test.shape)     # (10000,)
# y_train = y_train.reshape(y_train[0],1)
print(x_train.shape[0])


np.save('./data/mnist_train_x.npy',arr=x_train,)
np.save('./data/mnist_train_y.npy',arr=y_train)
np.save('./data/mnist_test_x.npy',arr=x_test)
np.save('./data/mnist_test_y.npy',arr=y_test)

