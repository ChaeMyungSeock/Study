
# 1. 데이터

import numpy as np

x1 = np.array([range(1,101), range(311, 411)])
y1 = np.array([range(711,811), range(711,811)])

x2 = np.array([range(101,201), range(411, 511)])
y2 = np.array([range(501,601), range(711,811)])

y3 = np.array([range(411,511), range(611,711)])

######################################
############## 여기서부터 수정 ########
######################################
x1= x1.transpose()
y1= y1.transpose()



x2= x2.transpose()
y2= y2.transpose()

y3= y3.transpose()

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, shuffle = False,  train_size = 0.8 )
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, shuffle = False,  train_size = 0.8 )
y3_train, y3_test = train_test_split(y3, shuffle = False,  train_size = 0.8 )


# (x,y, random_state = 66, 
   
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
input1 = Input(shape = (2,) , name='start') #열우선
dense1 = Dense(25, activation='relu', name ='start2' )(input1)
dense1 = Dense(35, activation='relu',name ='start3')(dense1)
dense1 = Dense(45, activation='relu',name ='start4')(dense1)
output1 = Dense(50)(dense1)

input2 = Input(shape = (2,)) #열우선
dense2 = Dense(65, activation='relu')(input1)
dense2 = Dense(75, activation='relu')(dense2)
dense2 = Dense(85, activation='relu')(dense2)
output2 = Dense(90)(dense2)

merge1 = concatenate([output1, output2])
# merge1 = Concatenate()[output1, output2]

model1 = Dense(10)(merge1)
model1 = Dense(11)(model1)

############### output 모델 구성

output3 = Dense(31)(model1)
output3_1 = Dense(2)(output3)

output4 = Dense(32)(model1)
output4_1 = Dense(2)(output4)

output5 = Dense(32)(model1)
output5_1 = Dense(2)(output5)

model1 = Model(inputs = [input1, input2], outputs = [output3_1, output4_1, output5_1])








model1.summary()



#3. 훈련
model1.compile(loss = 'mse', optimizer = 'adam', metrics=['mse']) # acc분류 지표 따라서 오차가 발생함에도 acc 1이 나옴
model1.fit([x1_train,x2_train] ,
            [y1_train,y2_train, y3_train],epochs=80, batch_size=1, validation_split=0.25,verbose=1)



#epochs를 2000으로 fit을 했을 때 중간에 loss가 감소하다가 증가하는 구간이 발생 why? overfiting

x_test = [x1_test, x2_test]
y_test = [y1_test, y2_test,y3_test]

#4. 평가,예측
loss_tot, loss1, loss2 ,loss3,mse1, mse2, mse3 = model1.evaluate(x_test,y_test, batch_size=1)



# 이미 훈련한 데이터로 평가를 할 때 다시 같은 데이터를 입력하게 되면 당연한 결과값이 나옴 (이미 결과값을 알고 있음)

print ("loss1 : ", loss1)
print ("mse1 : ", mse1)

print ("loss2 : ", loss2)
print ("mse2 : ", mse2)

print ("loss3 : ", loss2)
print ("mse3 : ", mse2)

print ("loss_tot : ", loss_tot)

# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)

y1_predict, y2_predict, y3_predict = model1.predict([x1_test,x2_test]) # 20by3=>test 2개
print(y1_predict)
print("=====================")
print(y2_predict)
print("=====================")
print(y3_predict)




# RMSE 구하기
from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용

def RMSE(y_test ,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)
RMSE3 = RMSE(y3_test,y3_predict)

print("RMSE : ", (RMSE1+RMSE2+RMSE3)/3)
print("RMSE_final : ", np.sqrt(mse1 + mse2 + mse3)/3)

# R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)

print("R2 : ",(r2_1 + r2_2 + r2_3)/3 )
# print(model1.metrics_names)
