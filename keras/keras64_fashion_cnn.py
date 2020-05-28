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

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],1)

y_train = np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
print(x_train.shape)

# minmax의 경우 2차원만 가능하므로 데이터를 2차원으로 떨군뒤에 다시 데이터 차원을 올려줘야함

# print(x_train[0])
# print('y_train[0] : ', y_train[0])

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 2. 모델
model = Sequential()
model.add(Conv2D(100, (3,3), input_shape = (28,28,1)))
model.add(Conv2D(300, (3,3), padding = 'same'))
model.add(Dropout(0.3))
model.add(Conv2D(500, (3,3), padding = 'same'))
model.add(Dropout(0.3))
model.add(Conv2D(300, (3,3), padding = 'same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


# 3. 훈련
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
earlystopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')
model.fit(x_train, y_train, batch_size=100, epochs=2, validation_split=0.1, callbacks=[earlystopping])


# 4. 평가, 예측

loss, acc = model.evaluate(x_train, y_train)

y_predict = model.predict(x_test)
print("loss : ", loss)
print("acc : ", acc)

print(y_predict)
print(y_test)


# loss :  0.06348036443367601
# acc :  0.9768146872520447