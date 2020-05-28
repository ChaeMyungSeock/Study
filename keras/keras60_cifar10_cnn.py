# keras60_cifar10_cnn.py

# 1. 데이터
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Input, Model
from keras.layers import Dense, LSTM
from keras.layers import Flatten,Conv2D,MaxPool2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2],3,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2],3,1)

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
input1 = Input(shape = (1024,3,1))
dense1 = Conv2D(32, (3,3), padding = 'same')(input1)
dense2 = Conv2D(32, (3,3), padding = 'same')(dense1)
dense3 = Dropout(0.3)(dense2)
dense4 = Conv2D(32, (3,3), padding = 'same')(dense3)
dense5 = Dropout(0.3)(dense4)
dense6 = Conv2D(32, (3,3), padding = 'same')(dense5)
dense7 = MaxPool2D(pool_size = 2)(dense6)
dense8 = Dropout(0.3)(dense7)
dense9 = Flatten()(dense8)
output1 = Dense(10,activation = 'softmax')(dense9)

model = Model(input = input1, output = output1)


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
