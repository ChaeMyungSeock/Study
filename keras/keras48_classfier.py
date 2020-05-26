import numpy as np

# 1. 데이터
x = np.array(range(1, 11))
y = np.array([1,0,1,0,1,0,1,0,1,0])
# (10,) => data 10개인 벡터 1개 dim = 1
# 이진분류가 필요한 데이터
# 
# def split_x(x):

from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
from keras.layers import Layer, Dense

x_train, x_test, y_train, y_test = train_test_split(x,y , train_size=0.8, shuffle = True, random_state = 66)

# 2. 모댈
model = Sequential()
model.add(Dense(500,activation='relu', input_shape = (1,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100))
model.add(Dense(1,activation='sigmoid'))

model.summary()

# 3.훈련
from keras.callbacks import EarlyStopping, TensorBoard

earlystopping = EarlyStopping(monitor ='loss', patience=10 , mode = 'min')
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['acc'])
# adam -> sigmoid, 시그모이드를 쓰는 이유는 0 or 1을 반환해줌
# 이진분류에서 사용하는 loss는 binary_crossentropy
model.fit(x,y,batch_size=1, epochs=1000, verbose=1, validation_split=0.2, callbacks=[earlystopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x,y,batch_size=1)
print("loss :",loss)
print("acc :",acc)

x_pre = np.array([1,2,3])
y_predict = model.predict(x_pre, batch_size=1)

print(y_predict)
print(y_predict.shape)
y_predict = y_predict.reshape(y_predict.shape[0])
print(y_predict.shape)


for i in range(len(y_predict)):
    y_predict[i] = round(y_predict[i])

print(y_predict)



# 과제 1 predict값을 0,1이 나오게 유도 sigmoid적용하는 함수 만듬, 누가 만든거 찾는다 무언가 다른게 있다