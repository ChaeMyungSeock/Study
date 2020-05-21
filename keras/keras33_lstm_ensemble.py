from numpy import array
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, GRU
from keras.layers.merge import concatenate
#1. 데이터
x1 = array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]
])
x2 = array([ [10,20,30], [20,30,40], [30,40,50], [40,50,60],
            [50,60,70], [60,70,80], [70,80,90], [80,90,100],
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]
])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (4,)

x_predict1 = array([55, 65, 75])
x_predict2 = array([65, 75, 85])

# x_predict2 = array([55, 65, 75])

print("x1.shape", x1.shape)   #(4 , 3)
print("y.shape", y.shape)   #(4 , ) 스칼라가 4개 input_dim = 1 => 1차원
print("x2.shape", x2.shape) #(1 , 4) 칼럼이 4개

x_predict1 = x_predict1.reshape(1,x_predict1.shape[0],1)
x_predict2 = x_predict2.reshape(1,x_predict2.shape[0],1)

x1 = x1.reshape(x1.shape[0], x1.shape[1],1)
x2 = x2.reshape(x2.shape[0], x2.shape[1],1)

# x = x.reshape(4,3,1)

print(x1.shape)
print(x2.shape)

'''
                행         열       몇개씩 자르는지
x의 shape = (batch_size, timestep, feature)
input_shape = (timesteps, feature)
input_length = temesteps, input_dim = feature


'''


#2. 모델 구성
input1 = Input(shape = (3,1))
dense1 = GRU(14, activation = 'relu')(input1)
dense1 = Dense(6)(dense1)

input2 = Input(shape = (3,1))
dense2 = GRU(14, activation = 'relu')(input2)
dense2 = Dense(6)(dense2)

middle1 = concatenate([dense1, dense2])
output1 = Dense(1)(middle1)

model = Model(input = [input1, input2], output = output1)

model.summary()


'''
model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape = (3,2))) # 실질적으로 행은 큰 영향을 안미치기 때문에 무시 몇개의 칼럼을 가지고 몇개씩 작업을 할 것인가
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(5000))
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(1))

model.summary()
'''

# 실행
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor= 'loss', patience=5, mode = 'min')
model.compile(optimizer='adam', loss = 'mse')
model.fit([x1,x2],y, epochs=10000, batch_size=5, callbacks=[earlystopping])





y_predict1= model.predict([x_predict1, x_predict2])
print(y_predict1)
# print(y_predict2)

