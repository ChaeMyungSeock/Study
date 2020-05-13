# 1. 데이터

import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])
# predict

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조에 가장 기본이 되는 Dense layer

model = Sequential()

model.add(Dense(50, input_dim = 1))
model.add(Dense(30))
# model.add(Dense(1000000)) 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(1))

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse']) # acc분류 지표 따라서 오차가 발생함에도 acc 1이 나옴
model.fit(x_train,y_train, epochs=2000, batch_size=1)

#4. 평가,예측
loss, mse = model.evaluate(x_test,y_test,batch_size=1)
# 이미 훈련한 데이터로 평가를 할 때 다시 같은 데이터를 입력하게 되면 당연한 결과값이 나옴 (이미 결과값을 알고 있음)

print ("loss : ", loss)
print ("mse : ", mse)



y_pred = model.predict(x_pred)

print("y_pred : ", y_pred)