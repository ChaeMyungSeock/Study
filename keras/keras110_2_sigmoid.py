import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,0,1,0,1,0,1,0,1,0])


# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input


model = Sequential()
model.add(Dense(100, input_shape = (1,)))
model.add(Dense(100))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))





model.summary()

# 3. 컴파일, 훈련
model.compile(loss = [ 'binary_crossentropy'], metrics=['acc'], optimizer='adam')

model.fit(x_train,[y_train],batch_size=1,epochs=100)

loss,acc = model.evaluate(x_train, [y_train])

print(loss)
print(acc)



x_pred = np.array([11,12,13,14])
y_pred = model.predict(x_pred)
print(y_pred)