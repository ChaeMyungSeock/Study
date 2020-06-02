# keras91 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import  np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

x_train = np.load('./data/mnist_train_x.npy')
x_test = np.load('./data/mnist_test_x.npy')
y_train = np.load('./data/mnist_train_y.npy')
y_test = np.load('./data/mnist_test_y.npy')

print('y_train : ', y_train.shape)
print(y_test.shape)
print(x_train.shape)    
print(x_test.shape)     



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





# 2. 모델
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (28,28,1)))
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Conv2D(32, (3,3), padding = 'same',))
model.add(MaxPool2D(pool_size=2)) # MaxPool 자원소모 x Conv2D + MaxPool2D 한 layer라고 생각하는게 편함
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()



# 3. 컴파일, 훈련
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc']) # metrics = ['acc']
# modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'    
earlystopping = EarlyStopping(monitor='loss',patience=3, mode='min')
# checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_weights_only=False, verbose = 1, save_best_only = True, mode = 'auto')     # save_best_only 좋은것만 저장, mode에 defalut값 존재
hist = model.fit(x_train, y_train, batch_size=200, epochs=10, validation_split=0.15, callbacks=[earlystopping]) 


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



plt.subplot(2, 1, 2)                # 2행 1열의 두번째 그림 index 1부터 n행 n열의 n번째 그림 (n, n, n)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()                      ``  # 눈금그려줌
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()



# 4. 평가 예측
loss_acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1) 
print("loss, acc : ", loss_acc)
# print("acc : ", acc)

print(y_predict)


# save 1
# loss :  125.75435492248535
# acc :  0.8805999755859375

# loss :  164.74859672555922
# acc :  0.8871999979019165
