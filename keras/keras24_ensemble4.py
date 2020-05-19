
# 1. 데이터

import numpy as np

x1 = np.array(range(1,101))

y1 = np.array([range(711,811), range(611,711)])
y2 = np.array([range(101,201), range(411,511)])


######################################
############## 여기서부터 수정 ########
######################################


x1= x1.transpose()

y1= y1.transpose()
y2= y2.transpose()





from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test  = train_test_split(x1,y1,y2, shuffle = False,  train_size = 0.8 )




   
#  x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 66, test_size = 0.4 )

#  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, shuffle = False, test_size = 0.5) # test_size의 default 값은 0.25



#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input #DNN구조에 가장 기본이 되는 Dense layer , Input layer => 함수형
from keras.layers.merge import concatenate
# model = Sequential()
# model.add(Dense(5, input_dim = 3))
# model.add(Dense(4))
# model.add(Dense(1))
input1 = Input(shape = (1,) , name='start') #열우선
dense1 = Dense(50, activation='relu', name ='start2' )(input1)
dense1 = Dense(35, activation='relu',name ='start3')(dense1)
dense1 = Dense(45, activation='relu',name ='start4')(dense1)
in_out = Dense(50)(dense1)

# merge1 = Concatenate()[output1, output2]


############### output 모델 구성

output1 = Dense(31)(in_out)
dense2 = Dense(2)(output1)

output2 = Dense(30)(in_out)
dense3 = Dense(2)(output2)


model1 = Model(inputs = input1, outputs = [dense2, dense3])




model1.summary()



#3. 훈련
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor = 'loss',patience =10,mode='min')
model1.compile(loss = 'mse', optimizer = 'adam', metrics=['mse']) # acc분류 지표 따라서 오차가 발생함에도 acc 1이 나옴
model1.fit(x1_train , [y1_train,y2_train],epochs=200, batch_size=1, validation_split=0.25,verbose=1, callbacks=[earlystopping])



#epochs를 2000으로 fit을 했을 때 중간에 loss가 감소하다가 증가하는 구간이 발생 why? overfiting


#4. 평가,예측
loss_tot, loss1, loss2, mse1, mse2 = model1.evaluate(x1_test, [y1_test, y2_test], batch_size=1)


# 이미 훈련한 데이터로 평가를 할 때 다시 같은 데이터를 입력하게 되면 당연한 결과값이 나옴 (이미 결과값을 알고 있음)

print ("loss : ", loss_tot, loss1, loss2)
print ("mse : ", mse1, mse2)


# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)

y1_predict , y2_predict = model1.predict(x1_test) # 20by3=>test 2개

y_predict = ([y1_predict,y2_predict])
# RMSE 구하기

from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용

def RMSE(y_test ,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE1 : ",RMSE(y1_test,y1_predict))
print("RMSE2 : ",RMSE(y2_test,y2_predict))


# R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)
r2_1 = r2_score(y2_test, y2_predict)

print("R2 : ",r2)
print("R2 : ",r2_1)

# print("R2 : ",(r2_1 + r2_2 + r2_3)/3 )
# print(model1.metrics_names)

