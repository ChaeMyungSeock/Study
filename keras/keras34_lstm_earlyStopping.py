from numpy import array
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split
#1. 데이터
x = array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]
])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (4,)
# y2 = array([[4,5,6,7]])     #(1 , 4)
# y3 = array([[4], [5], [6], [7]])   #(4,1)
print("x.shape", x.shape)   #(4 , 3)
print("y.shape", y.shape)   #(4 , ) 스칼라가 4개 input_dim = 1 => 1차원
# print("y2.shape", y2.shape) #(1 , 4) 칼럼이 4개
# print("y3.shape", y3.shape) #(4 , 1) 칼럼이 1개


x = x.reshape(x.shape[0], x.shape[1],1)

# x_train , x_test, y_train, y_test = train_test_split(x,y, train_size = 1, random_state = 66, shuffle = True)
# x = x.reshape(4,3,1)

print(x.shape)

#2. 모델 구성
# model = Sequential()
# # model.add(LSTM(10, activation = 'relu', input_shape = (3,2))) # 실질적으로 행은 큰 영향을 안미치기 때문에 무시 몇개의 칼럼을 가지고 몇개씩 작업을 할 것인가
# model.add(LSTM(130, input_length = 3, input_dim = 1))
# model.add(Dense(149, activation= 'sigmoid'))
# model.add(Dense(35))
# model.add(Dense(60))
# model.add(Dense(90))
# model.add(Dense(690))
# model.add(Dense(610))
# model.add(Dense(470))
# model.add(Dense(250))
# model.add(Dense(1))

input1 = Input(shape = (3,1))
dense1 = LSTM(20)(input1)
dense1 = Dense(420)(dense1)
output1 = Dense(1)(dense1)

model = Model(input = input1, output = output1)

model.summary()

# 실행
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor= 'loss', patience=10, mode = 'min')
model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs=1000, batch_size=10, callbacks=[earlystopping])

x_predict = array([50, 60, 70])
x_predict = x_predict.reshape(1,x_predict.shape[0],1)

print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
