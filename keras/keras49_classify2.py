import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.utils import  np_utils

# 1. 데이터
x = np.array(range(1, 11))
y = np.array([1,2,3,4,5,1,2,3,4,5])

print(y.shape)


y = np_utils.to_categorical(y) # 코드 공부

y = y[:,1:6]
print(y.shape)
print(y)


# 2. 모델

from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
from keras.layers import Layer, Dense
model = Sequential()
model.add(Dense(500,activation='relu', input_shape = (1,)))

model.add(Dense(5,activation='softmax'))
# softmax 다중분류 변화치에 따른 가중치 부여
model.summary()

# 3.훈련
from keras.callbacks import EarlyStopping, TensorBoard

earlystopping = EarlyStopping(monitor ='loss', patience=10 , mode = 'min')
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
# adam -> sigmoid, 시그모이드를 쓰는 이유는 0 or 1을 반환해줌
# 이진분류에서 사용하는 loss는 binary_crossentropy
model.fit(x,y,batch_size=1, epochs=100, verbose=1, validation_split=0.2, callbacks=[earlystopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x,y,batch_size=1)
print("loss :",loss)
print("acc :",acc)

x_pre = np.array([1,2,3,4,5])
y_predict = model.predict(x_pre, batch_size=1)
print(y_predict)

y_predict = np.argmax(y_predict,axis=1) + 1

print(y_predict)

# max(5,3) = 5

'''
값들이 들어올 때 전체적으로 선형적으로 받는 것이 아니라 각 행렬 안에서 True, False로 값을 선형적으로 바꿔주는것
[[0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]]

onehotencoding => 원하는 숫자를 제외한 모든걸 0으로 만들어 버려서 원하는 순서의 숫자만 encoding해서 가중치를 일정하게 가져감
'''





# import numpy as np
# from keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder
# # 1. 데이터
# x = np.array(range(1, 11))
# y = np.array([1,2,3,4,5,1,2,3,4,5])

# y_train= y.reshape(-1,1)
# enc = OneHotEncoder()
# enc.fit(y_train)
# y_train = enc.transform(y_train).toarray()

# print(y_train)
# print(y_train.shape)

# # 2. 모델
# from keras.models import Sequential, Model
# from keras.layers import Input,Dense

# model = Sequential()
# model.add(Dense(256, input_dim = 1, activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(5, activation='softmax'))

# # 3. 훈련
# from keras.callbacks import EarlyStopping
# earlystopping = EarlyStopping(monitor='loss',patience=2, mode= 'min')
# model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics=['acc'])
# model.fit(x,y_train, batch_size=1, epochs=1000, callbacks=[earlystopping])


# # 4. 평가, 예측

# loss, acc = model.evaluate(x,y_train,batch_size=1)

# x_pre = np.array([1,2,3,4,5])
# y_predict = model.predict(x_pre, batch_size=1)
# y_predict = np.argmax(y_predict,axis=1).reshape(-1,)

# print(y_predict)
