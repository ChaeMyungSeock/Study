import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])


# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

x2 = Dense(50)(x1)
output1= Dense(1)(x2)


x3 = Dense(70)(x1)
x3 = Dense(70)(x3)
output2 = Dense(1, activation='sigmoid')(x3)

model = Model(inputs = input1, outputs = [output1, output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], metrics=['mse', 'acc'], optimizer='adam')

model.fit(x_train,[y1_train,y2_train],batch_size=1,epochs=100)

loss, loss1, loss2, mse1, acc1, mse2, acc2 = model.evaluate(x_train, [y1_train,y2_train])

print(loss)
print(loss1)
print(loss2)
print("=========")
print(mse1)
print(mse2)
print("=========")
print(acc1)
print(acc2)

x_pred = np.array([11,12,13,14])
y_pred = model.predict(x_pred)
print(y_pred)