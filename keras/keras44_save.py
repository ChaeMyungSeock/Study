# keras44_save.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


# 2. 모델
model = Sequential()
model.add(LSTM(24,input_shape = (4,1)))
model.add(Dense(380))
model.add(Dense(10))
model.summary()

# model.save(".//model//save_keras44.h5") #경로를 지정해주지 않으면 경로 기본폴더에 저장
# model.save(".\model\save_keras44.h5") #경로를 지정해주지 않으면 경로 기본폴더에 저장
model.save("./model/save_keras44.h5") #경로를 지정해주지 않으면 경로 기본폴더에 저장

print("저장 됨")