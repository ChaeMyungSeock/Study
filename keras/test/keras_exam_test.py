from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import concatenate
import numpy as np
import pandas as pd


# 1. 데이터
data_samsung_load = np.load('./data/samsung_0602.npy', allow_pickle= True)
data_hite_load = np.load('./data/hite0602.npy', allow_pickle= True)


# print(data_samsung_load.__class__)
# print(data_hite_load.__class__)


# print(data_hite_load.shape)
# print(data_samsung_load.shape)


hite_puls = np.array([39000,40108,38321,38695,513031])




# print(hite_puls.shape)
hite_puls = hite_puls.reshape(1,hite_puls.shape[0])
data_hite_load = np.append(data_hite_load, hite_puls, axis=0)

# print(data_hite_load.shape)


def split_xy(dataset, time_steps, y_column):
    x,y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number:y_end_number,:]

        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1,y1 = split_xy(data_hite_load,5,1)
x2,y2 = split_xy(data_samsung_load,5,1)



x1_train, x1_test, y1_train, y1_test= train_test_split(x1,y1, test_size=0.2,shuffle =False)
x2_train, x2_test, y2_train, y2_test= train_test_split(x2,y2, test_size=0.2, shuffle= False)

# print(x1_train.shape)       # (453, 5, 5)
# print(x1_test.shape)        # (51, 5, 5)
# print(x2_train.shape)       # (453, 5, 1)
# print(x2_test.shape)        # (51, 5, 1)

# print(y1_train.shape)       # (453, 1, 5)
# print(y1_test.shape)        # (51, 1, 5)
# print(y2_train.shape)       # (453, 1, 1)
# print(y2_test.shape)        # (51, 1, 1)


x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])


y1_train = y1_train.reshape(y1_train.shape[0], y1_train.shape[1]*y1_train.shape[2])
y1_test = y1_test.reshape(y1_test.shape[0], y1_test.shape[1]*y1_test.shape[2])


x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2])


y2_train = y2_train.reshape(y2_train.shape[0], y2_train.shape[1]*y2_train.shape[2])
y2_test = y2_test.reshape(y2_test.shape[0], y2_test.shape[1]*y2_test.shape[2])

scaler = StandardScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
scaler.fit(x1_test)
x1_test = scaler.transform(x1_test)

scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)

scaler.fit(x2_test)
x2_test = scaler.transform(x2_test)

scaler.fit(y1_train)
y1_train = scaler.transform(y1_train)

scaler.fit(y1_test)
y1_test = scaler.transform(y1_test)


scaler.fit(y2_train)
y2_train = scaler.transform(y2_train)

scaler.fit(y2_test)
y2_test = scaler.transform(y2_test)

# x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
# x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)


# x2_train = x2_train.reshape(x2_train.shape[0], 5, 1)
# x2_test = x2_test.reshape(x2_test.shape[0], 5, 1)

print(x1_train.shape)       # (453, 5, 5)
print(x1_test.shape)        # (51, 5, 5)
print(x2_train.shape)       # (453, 5, 1)
print(x2_test.shape)        # (51, 5, 1)


# 2. 모델

input1 = Input(shape = (25,) , name='start') #열우선
dense1 = Dense(300, name ='start2',activation='relu' )(input1)
dense1 = Dropout(0.2)(dense1)
output1 = Dense(50,activation='relu')(dense1)

input2 = Input(shape = (5,)) #열우선
dense2 = Dense(300, activation='relu' )(input1)
dense2 = Dropout(0.2)(dense2)
output2 = Dense(50,activation='relu')(dense2)

merge1 = concatenate([output1, output2])
# merge1 = Concatenate()[output1, output2]

model1 = Dense(3000,activation='relu')(merge1)

############### output 모델 구성

output3_1 = Dense(5)(model1)


output4_1 = Dense(1)(model1)

model1 = Model(inputs = [input1, input2], outputs = [output3_1, output4_1])

# 3. 컴파일 및 훈련
model1.compile(loss = 'mse', optimizer = 'rmsprop', metrics=['mse']) # acc분류 지표 따라서 오차가 발생함에도 acc 1이 나옴
earlystopping = EarlyStopping(patience=5, mode='min',monitor='loss')
model1.fit([x1_train,x2_train] ,[y1_train,y2_train],epochs=20, shuffle=True, batch_size=1, validation_split=0.2,verbose=1)

model1.save_weights('./model/test_0602weight1.h5')
# 4. 평가 및 예측
loss, mse, loss1, mse1, loss2 = model1.evaluate([x1_test,x2_test], [y1_test, y2_test], batch_size=1)

# print("loss : ", loss)
# print("mse : ", mse)



y1_predict, y2_predict = model1.predict([x1_test, x2_test])


y2_predict_1 = scaler.inverse_transform(y2_predict)


print('삼성주가', y2_predict_1)
# RMSE 구하기
from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용

def RMSE(y_test ,y_predict1):
    return np.sqrt(mean_squared_error(y1_test,y1_predict))
RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)

    # print("RMSE : ", (RMSE1+RMSE2)/2)

# R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2 = r2_1 + r2_2
   
print("R2 : ",(r2_1 + r2_2)/2 )
        # print('하이트주가', y1_predict)
        # print('삼성주가', y2_predict)


# predict로


