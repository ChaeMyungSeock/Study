# keras35_lstm_sequences.py

from numpy import array
from keras.models import Model,Sequential
from keras.layers import Input, LSTM, Dense, Flatten # Flatten() 의 경우 3차원을 2차원으로 강제 데이터 혹은 차원을 바꿔줌
from sklearn.model_selection import train_test_split
#1. 데이터
x = array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [2000,3000,4000], [3000,4000,5000], [4000,5000,6000],
            [100, 200, 300]
])
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000, 400]) # (4,)

# print("x.shape", x.shape)   #(14 , 3)
# print("y.shape", y.shape)   #(14 , ) 스칼라가 4개 input_dim = 1 => 1차원


x_predict = array([55, 65, 75])
print(x.shape)
print(x_predict.shape)
x_predict = x_predict.reshape(1,3)
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x) # => fit 실행하다 MinMaxScaler를 실행해라
x=scaler.transform(x) # 실행한 MinMaxScaler를 변환해줘라
x_predict = scaler.transform(x_predict)

print(x)
print(x_predict)


x = x.reshape(x.shape[0], x.shape[1],1)

# x_train , x_test, y_train, y_test = train_test_split(x,y, train_size = 1, random_state = 66, shuffle = True)
# x = x.reshape(4,3,1)


x_predict = x_predict.reshape(1,x_predict.shape[1],1)


#2. 모델 구성
model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_shape. = (3,2))) # 실질적으로 행은 큰 영향을 안미치기 때문에 무시 몇개의 칼럼을 가지고 몇개씩 작업을 할 것인가
model.add(LSTM(10, input_length = 3, input_dim = 1, return_sequences = False))
# Dense layer는 행열 2차원만 받고 output 또한 2차원 하지만 LSTM (행, 열, 피쳐) 3차원을 필요로 함으로 return_seauences를 리턴해줘서 
model.add(Dense(100))
model.add(Dense(1))

# input1 = Input(shape = (3,1))
# dense1 = LSTM(20)(input1)
# dense1 = Dense(420)(dense1)
# output1 = Dense(1)(dense1)

# model = Model(input = input1, output = output1)

model.summary()

# 실행
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor= 'loss', patience=20, mode = 'min')
model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs=1000, batch_size=10, callbacks=[earlystopping])


print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
