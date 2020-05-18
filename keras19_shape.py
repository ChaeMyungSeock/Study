
# 1. 데이터

import numpy as np

x = np.array([range(1,101), range(311, 411), range(100)])
y = np.array([range(711,811)])


x= np.transpose(x)
y=np.transpose(y)
print(x.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle = False,  train_size = 0.8 )
# (x,y, random_state = 66, 
   
#  x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 66, test_size = 0.4 )

#  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, shuffle = False, test_size = 0.5) # test_size의 default 값은 0.25



#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input #DNN구조에 가장 기본이 되는 Dense layer

# model = Sequential()

# model.add(Dense(5, input_dim = 3))
input1 = Input(shape = (3,)) #열우선
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
output1 = Dense(1)(dense1)


model = Model(input = input1, output = output1)
model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse']) # acc분류 지표 따라서 오차가 발생함에도 acc 1이 나옴
model.fit(x_train,y_train, epochs=70, batch_size=1, validation_split=0.25,verbose=2)
#epochs를 2000으로 fit을 했을 때 중간에 loss가 감소하다가 증가하는 구간이 발생 why? overfiting


#4. 평가,예측
loss, mse = model.evaluate(x_test,y_test,batch_size=1)
# 이미 훈련한 데이터로 평가를 할 때 다시 같은 데이터를 입력하게 되면 당연한 결과값이 나옴 (이미 결과값을 알고 있음)

print ("loss : ", loss)
print ("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict) 

# RMSE 구하기
from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

