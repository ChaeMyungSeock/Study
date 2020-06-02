# keras53 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import  np_utils

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28) batch_size = 60000 28 * 28 이미지
print(x_test.shape)     # (10000, 28, 28) batch_size = 10000 28 * 28
print(y_train.shape)    # (60000,)  inputdim = 1
print(y_test.shape)     # (10000,)
# y_train = y_train.reshape(y_train[0],1)
print(x_train.shape[0])

x_train = x_train / 255
# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2],1))

y_train = np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
print('y_train : ', y_train.shape)

print(x_train.shape)    
print(x_test.shape)     
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D

# 2. 모델

# model = Sequential()
# model.add(Con))
# model.add(Flatten())
# model.add(Dense(1))
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (28,28,1)))
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Conv2D(32, (3,3), padding = 'same',))
model.add(MaxPool2D(pool_size=2)) # MaxPool 자원소모 x Conv2D + MaxPool2D 한 layer라고 생각하는게 편함
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

# model.save('./model/model_test01.h5')

# 3. 훈련
from keras.callbacks import EarlyStopping
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc']) # metrics = ['acc']
earlystopping = EarlyStopping(monitor='loss',patience=3, mode='min')
hist = model.fit(x_train, y_train, batch_size=200, epochs=20, validation_split=0.15, callbacks=[earlystopping]) 

model.save('./model/model_test01.h5')


import matplotlib.pyplot as plt


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss: ', loss)
print('val_loss: ', val_loss)




plt.figure(figsize=(20,10))      # 10인치 * 6인치

plt.subplot(2, 1, 1)            # 2행 1열의 첫번째 그림

plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')              # maker 점을 찍어줌 색깔 
plt.plot(hist.history['val_loss'], marker = '.', c ='blue', label = 'val_loss')
# plt.plot(x,y)
plt.plot(hist.history['val_loss'])
plt.grid()                      # 눈금그려줌
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()



plt.subplot(2, 1, 2)            # 2행 1열의 두번째 그림 index 1부터 n행 n열의 n번째 그림 (n, n, n)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()                      # 눈금그려줌
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()


# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1) 
print("loss : ", loss)
print("acc : ", acc)

print(y_predict)


# save 1
# loss :  125.75435492248535
# acc :  0.8805999755859375