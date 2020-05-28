import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Input, Model
from keras.layers import Dense, LSTM
from keras.layers import Flatten,Conv2D,MaxPool2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
# 과제 1

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[3])
plt.show()