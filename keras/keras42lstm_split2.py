#keras42_lstm_split2.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
# 1. 데이터

a = np.array(range(1,101 ))
print(a)
size = 5                    # time_steps = 5

def split_x (x_list, size):
    x_list = []
    for i in range(len(a) - size +1 ):
        xset = a[i:size+i]
        x_list.append(xset)
    return np.array(x_list)

# def split_y (y_list, size):
#     y_list = []
#     for j in range(len(a) - size ):
#         yset = a[size + j]
#         y_list.append(yset)
#     return np.array(y_list)

# data_x = split_x(a,5)
# data_y = split_y(a,5)

# print(data_x.shape)
# print(data_y.shape)

# print(data_x)
# print(data_y)


# data_x = data_x.reshape(data_x.shape[0],data_x.shape[1],1)

dataset = split_x(a, size)      #(6,5)
print(dataset)

x1 = dataset[:, 0:4]
y1 = dataset[:, 4]
# x1_predict = x1[90:96 ]
# x_train = x1[0:90]
# # y_predict = y1[90:96 ]
# y_train = y1[0:90]


# x1 = x1.reshape(x1.shape[0], x1.shape[1],1)
# x1 = np.reshape(x1, (6,4,1))
# print(x_train.shape)

# 실습 1. train, test 분리
# 실습 2. 마지막 6개의 행을 predict로 만들고 싶다
# 실습 3. validation을 넣을 것 (train의 20%)

x1_train,x1_predict ,y1_train, y1_predict = train_test_split(x1, y1,shuffle = True, train_size=90/96, random_state = 66 )


x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1],1)
x1_predict = x1_predict.reshape(x1_predict.shape[0], x1_predict.shape[1],1)



# LSTM 모델을 만드시오.
# 2. 모델
model = Sequential()
model.add(LSTM(10,input_shape = (4,1)))
model.add(Dense(500))
model.add(Dense(1))
model.summary()


# 3. 실행
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.compile(optimizer = 'adam', loss = 'mse', metrics=['mse'])
model.fit(x1_train, y1_train, validation_split=0.2 ,epochs=1000, callbacks=[earlystopping], batch_size=1, verbose=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x1_train, y1_train, batch_size=1)

print("loss : ", loss)
print("mse : ", mse)


x = model.predict(x1_predict)
print("predict : ",x)
print(y1_predict)


