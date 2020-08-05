# keras56_mnist_dnn.py 복붙
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28) batch_size = 60000 28 * 28 이미지
print(x_test.shape)     # (10000, 28, 28) batch_size = 10000 28 * 28
print(y_train.shape)    # (60000,)  inputdim = 1
print(y_test.shape)     # (10000,)
# y_train = y_train.reshape(y_train[0],1)
print(x_train.shape[0])
x_train = x_train / 255
x_test = x_test / 255

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]*1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]*1))

# y_train = np_utils.to_categorical(y_train)
# y_test= np_utils.to_categorical(y_test)
# print('y_train : ', y_train.shape)

print(x_train.shape)    
print(x_test.shape)     
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

X = np.append(x_train, x_test, axis=0)

print(X.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum =np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)

n_components = np.argmax(cumsum>=0.95)
# print(cumsum>=0.94)
print(n_components+1)