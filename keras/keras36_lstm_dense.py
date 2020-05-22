# keras36_lstm_dense.py
import os
from numpy import array
from keras.models import Model,Sequential
from keras.layers import Input, LSTM, Dense, Flatten,Reshape # Flatten() 의 경우 3차원을 2차원으로 강제 데이터 혹은 차원을 바꿔줌
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


x = x.reshape(x.shape[0], x.shape[1])

# x_train , x_test, y_train, y_test = train_test_split(x,y, train_size = 1, random_state = 66, shuffle = True)
# x = x.reshape(4,3,1)

print(x.shape)

#2. 모델 구성
model = Sequential()
# input1 = Input(input_shape = (3,1))
# model.add(LSTM(10, activation = 'relu', input_shape = (3,2))) # 실질적으로 행은 큰 영향을 안미치기 때문에 무시 몇개의 칼럼을 가지고 몇개씩 작업을 할 것인가
model.add(Dense(10,input_shape=(3,)))
# model.add(Dense(10,Reshape((3,),input_shape=(3,)))

# Dense layer는 행열 2차원만 받고 output 또한 2차원 하지만 LSTM (행, 열, 피쳐) 3차원을 필요로 함으로 return_seauences를 리턴해줘서 
# model.add(LSTM(10,return_sequences =False))
model.add(Dense(10))
model.add(Dense(35))

model.add(Dense(1))

# input1 = Input(shape = (3,1))
# dense1 = LSTM(20)(input1)
# dense1 = Dense(420)(dense1)
# output1 = Dense(1)(dense1)

# model = Model(input = input1, output = output1)

model.summary()



'''
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

'''