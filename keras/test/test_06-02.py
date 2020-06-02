from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# 1. 데이터
data_samsung_load = np.load('./data/samsung_0602.npy', allow_pickle= True)
data_hite_load = np.load('./data/hite0602.npy', allow_pickle= True)


print(data_hite_load)
# print(data_samsung_load)

# hite의 고가 저가 종가 거래량을 predict
                                                       
# 다데이터 1입력 다 결과

def split_xy(dataset, time_steps, y_column):
    x,y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number:y_end_number,4]

        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

print('=================')
x0_train = np.array(data_hite_load[:508,0])
y0_train = np.array(data_hite_load[:508,1:5])

x0_test = np.array([39000])
print(x0_train.shape)

# x0_train = x0_train.reshape(x0_train.shape[0],1)
# x0_test = x0_test.reshape(x0_test.shape[0],1)
# x0_r_test = x0_r_test.reshape(x0_r_test.shape[0],1)

# print(y0_train.shape)

model = Sequential()

model.add(Dense(100, input_shape = (1,),activation='relu'))
model.add(Dense(3000,activation='relu'))
model.add(Dense(4))
model.summary()

for i in range(50):

    #3. 훈련
    model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse']) 
    model.fit(x0_train,y0_train, epochs=i+1, batch_size=1, validation_split=0.2,verbose=0)
    #4. 평가,예측
    # loss, mse = model.evaluate(x0_test,y0_test,batch_size=1)

    # print ("loss : ", loss)
    # print ("mse : ", mse)

    # y_pred = model.predict(x_pred)
    # print("y_pred : ", y_pred)

    y_predict = model.predict(x0_test)
    if(y_predict[0,1]<=39000):
        print('y_predict : ',y_predict) 
        y_predict = np.around(y_predict)
        print(y_predict)
        
    
# # RMSE 구하기
# from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test,y_predict))
# print("RMSE : ", RMSE(y0_test, y_predict))

# # R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
# from sklearn.metrics import r2_score
# r2 = r2_score(y0_test, y_predict)
# print("R2 : ", r2)

# [ 38191.64   39183.402  36792.887 488506.7  ]