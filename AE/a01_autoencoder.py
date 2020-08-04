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
print(x_train.__class__)
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


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
from tensorflow.keras.layers import Dropout, Input


# 2. 모델

input_img = Input(shape=(784,))
encoded = Dense(32, activation = 'relu')(input_img)
decoded = Dense(784,activation = 'sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs = 50, batch_size = 256,
                validation_split = 0.2)

encoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(encoded_imgs[i].reshape(28,28))
    plt.gray()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


'''
# 3. 훈련
from tensorflow.keras.callbacks import EarlyStopping
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss',patience=10, mode='min')
model.fit(x_train, y_train, batch_size=100, epochs=20, validation_split=0.1, callbacks=[earlystopping])

# 4. 평가 예측

loss, acc = model.evaluate(x_train, y_train)
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict,axis=1) 
print("loss : ", loss)
print("acc : ", acc)

# # print(y_predict)

# Test = pd.read_csv('C:\\Users\\bitcamp\Desktop\kaggle\\test.csv\\test.csv',engine='python',encoding='euc-kr')
# Test = Test / 255.0
# Test = Test.values.reshape(-1,28,28,1)

# res = model.predict(Test)
# res = np.argmax(res, axis = 1)
# res = pd.Series(res, name = 'Label')

# submit = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), res], axis = 1)
# submit.to_csv("C:\\Users\\bitcamp\Downloads\sample_submission.csv", index=False)

# submit.to_csv("mySubmission_mnist_cnn.csv", index = False)
'''