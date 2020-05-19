#keras14_mlp Sequential에서 함수형으로 변경
#earlyStopping 적용
# multi layer perceptron


import numpy as np

x = np.array([range(1,101), range(311, 411), range(100)])
y = np.array([range(101,201), range(711,811), range(100)])

x = x.transpose()
y = y.transpose()

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split # 선언하는거 정리 기억이 안남...

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state = 66)

#2. 모델구성
from keras.models import Input, Model
from keras.layers import Dense 


input1 = Input(shape=(3,))
dense1 = Dense(30, activation= 'relu')(input1)
dense1 = Dense(50, activation= 'relu')(dense1)
dense1 = Dense(100, activation= 'relu')(dense1)
dense1 = Dense(50, activation= 'relu')(dense1)
dense1 = Dense(50, activation= 'relu')(dense1)
dense1 = Dense(30, activation= 'relu')(dense1)
output1 = Dense(3)(dense1)

model = Model(input = input1 , output = output1)

model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping 
model.compile(loss= 'mse', optimizer='adam', metrics=['mse'])
earlyStopping = EarlyStopping(monitor = 'loss',patience =3, mode='min' )
model.fit([x_train],[y_train], batch_size=1, epochs=1000,verbose=1, callbacks=[earlyStopping] , validation_split=0.25)

#4. 평가,예측

loss, mse = model.evaluate(x_test, y_test, batch_size = 1, )

print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)

print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용

def RMSE(y_test ,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))



# R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)
