import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Input, Model, Sequential
from keras.layers import Dense, LSTM
from keras.layers import Flatten,Conv2D,MaxPool2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
# 과제 1

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train / 255
x_test = x_test / 255

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]*1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]*1))

y_train = np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
print('y_train : ', y_train.shape)

print(x_train.shape)    
print(x_test.shape)     
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


# 2. 모델

# model = Sequential()
# model.add(Con))
# model.add(Flatten())
# model.add(Dense(1))
model = Sequential()
# model.add(Conv2D(100, (3,3), input_shape = (28,28,1)))
# # 784 , 28*28
# model.add(Conv2D(300, (3,3), padding = 'same'))
# model.add(Dropout(0.3))
# model.add(Conv2D(500, (3,3), padding = 'same'))
# model.add(Dropout(0.3))
# model.add(Conv2D(300, (3,3), padding = 'same'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

# model.add(LSTM(10, input_shape = (784,1)))
model.add(Dense(30, input_shape = (784,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.add(Dense(10))

model.summary()

# 3. 훈련
from keras.callbacks import EarlyStopping
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
earlystopping = EarlyStopping(monitor='loss',patience=10, mode='min')
model.fit(x_train, y_train, batch_size=100, epochs=5, validation_split=0.1, callbacks=[earlystopping])

# 4. 평가 예측

loss, acc = model.evaluate(x_train, y_train)
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict,axis=1) 
print("loss : ", loss)
print("acc : ", acc)


# loss :  0.3738133368293444
# acc :  0.8673833608627319
