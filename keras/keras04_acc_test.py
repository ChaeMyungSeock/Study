# 1. 데이터

import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])
# predict

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조에 가장 기본이 되는 Dense layer

model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
# model.add(Dense(1000000)) 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc']) # acc분류 지표 따라서 오차가 발생함에도 acc 1이 나옴
                                                                 #
model.fit(x,y, epochs=30, batch_size=3)

#4. 평가,예측
loss, acc = model.evaluate(x,y,batch_size=3)

print ("loss : ", loss)
print ("acc : ", acc)

y_pred = model.predict(x_pred)

print("y_pred : ", y_pred)