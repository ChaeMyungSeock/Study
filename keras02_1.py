#51p
import numpy as np
#데이터 생성
from keras.models import Sequential
from keras.layers import Dense

x_train = np.array([2,3,4,5,6,7,8,9,10,11])
y_train = np.array([4,9,16,25,36,49,64,81,100,121])
x_test = np.array([10,11,12,13,14,15,16,17,18,19])
y_test = np.array([100,121,144,169,196,225,256,289,324,361])

model = Sequential()
model.add(Dense(600, input_dim=1, activation='relu')) #모델.add 담을 쌓겠다
model.add(Dense(500))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train,y_train, epochs=5000, batch_size=100,validation_data=(x_train,y_train))
loss, acc = model.evaluate(x_test,y_test, batch_size=100)

print("loss : ",loss)
print("acc : ", acc)

y_predict = model.predict(x_test)
print("결과물 : \n", y_predict)

