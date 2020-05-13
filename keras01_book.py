#p30
#하이퍼 파라미터
#1.노드갯수
#2.레이어의 깊이 Deep
#3.epochs
#4.batchsize
import numpy as np
#데이터 생성

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

print(x)
print(y)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x,y, epochs=500, batch_size=1)
loss, acc = model.evaluate(x,y, batch_size=1)

print("loss : ",loss)
print("acc : ", acc)

