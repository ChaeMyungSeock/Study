from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6]])
y = array([4,5,6,7])
y2 = array([[4,5,6,7]])     #(1 , 4)
y3 = array([[4], [5], [6], [7]])   #(4,1)
print("x.shape", x.shape)   #(4 , 3)
print("y.shape", y.shape)   #(4 , ) 스칼라가 4개 input_dim = 1 => 1차원
print("y2.shape", y2.shape) #(1 , 4) 칼럼이 4개
print("y3.shape", y3.shape) #(4 , 1) 칼럼이 1개


x = x.reshape(x.shape[0], x.shape[1],1)
# x = x.reshape(4,3,1)

print(x.shape)

#2. 모델 구성
model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape = (3,2))) # 실질적으로 행은 큰 영향을 안미치기 때문에 무시 몇개의 칼럼을 가지고 몇개씩 작업을 할 것인가
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(5000))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.summary()
'''
# 실행
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor= 'loss', patience=10, mode = 'min')
model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs=50, batch_size=1, callbacks=[earlystopping])

x_input = array([5, 6, 7])
x_input = x_input.reshape(1,x_input.shape[0],1)

print(x_input)

yhat = model.predict(x_input)
print(yhat)
'''