from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000, test_split=0.2)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))

# 카테고리 개수 출력
category = np.max(y_train)+1
print("카테고리 : ", category)

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train = pd.DataFrame(y_train)

print(y_train)
bbb = y_train.groupby(0)[0].count()

print(bbb)
print(bbb.shape)

'''
# 주간 과제 : groupby() 사용법 숙지

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

# x_train = pad_sequences(x_train)

# truncating 자르겠다

# print(len(x_train[0]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape , x_test.shape)


# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
model.add(Embedding(1000,100))
model.add(LSTM(100))
model.add(Dense(46, activation='softmax'))

# model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]

print('acc', acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''