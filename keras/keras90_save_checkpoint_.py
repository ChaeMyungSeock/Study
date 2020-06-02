# keras53 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import  np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
import matplotlib.pyplot as plt



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

# model.save('./model/model_test01.h5')

# 3. 컴파일, 훈련
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc']) # metrics = ['acc']
modelpath = './model/check{epoch:02d}-{val_loss:.4f}.hdf5'    
earlystopping = EarlyStopping(monitor='loss',patience=3, mode='min')
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_weights_only=False, verbose = 1, save_best_only = True, mode = 'auto')     # save_best_only 좋은것만 저장, mode에 defalut값 존재
hist = model.fit(x_train, y_train, batch_size=200, epochs=7, validation_split=0.15, callbacks=[earlystopping, checkpoint]) 

model.save('./model/model_test01.h5')








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

# loss :  164.74859672555922
# acc :  0.8871999979019165