# 1.데이터

import numpy as np
x_train = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
y_train = np.array([6,7,8])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)

# RNN은 순차형(sequential) 데이터를 모델링하는데 최적화된 구조

# print(x_train.shape) (3,5,1)
# print(y_train.shape) (3,)

# print(x_train)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
# from keras.layers import Dense, SimpleRNN


model = Sequential()

model.add(Bidirectional(LSTM(10,activation = 'relu'),input_shape = (5,1)))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.summary() 

# model.add(SimpleRNN(7, input_shape = (5,1), activation = 'relu'))
# model.add(SimpleRNN(30, input_length = 5 , input_dim=1,activation = 'relu'))
# model.add(Dense(150))
# model.add(Dense(1))

# model.summary()

'''
# 3. 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=60, batch_size = 1)


# 4. 예측

x_predict = np.array([[4,5,6,7,8]])
print(x_predict.shape)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print("x_predict.shape : ", x_predict.shape)

y_predict = model.predict(x_predict)
print(y_predict)
'''