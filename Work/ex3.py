from efficientnet import model
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Conv2D,MaxPool2D, Dropout,Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tfkeras import EfficientNetB0
from tfkeras import EfficientNetB1
from tfkeras import EfficientNetB2

from efficientnet import weights
# 과제 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print('시작')
x_train = np.load('D:/study/data/train_val.npy')
y_train = np.loadtxt('D:/study/data/train_val_label.txt')

x_test = np.load('D:/study/data/test.npy')
y_test = np.loadtxt('D:/study/data/test_label.txt')
x_test = x_test[:5138,:,:,:]
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)
print('끝')

# 2. 모델
model = Sequential()
model.add(EfficientNetB0(include_top=False))
model.add(Dense(20, activation='softmax',name='s1'))
model.summary()
# 3. 훈련
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
model.fit(x_train, y_train, batch_size=12, epochs=2, validation_split=0.1)


# 4. 평가, 예측

loss, acc = model.evaluate(x_train, y_train)

y_predict = model.predict(x_test)
print("loss : ", loss)
print("acc : ", acc)

print(y_predict)
print('진짜 끝')



