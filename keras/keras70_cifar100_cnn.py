from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Input, Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout,Dense, LSTM, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
# 1.데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

scaler.fit(x_test)
x_test = scaler.transform(x_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


# 2. 모델

input1 = Input(shape = (32,32,3))
dense1 = Conv2D(100, (3,3), padding = 'same')(input1)
dense2 = Conv2D(150, (3,3), padding = 'same',activation = 'relu')(dense1)
dense2 = Dropout(0.3)(dense2)
dense3 = Conv2D(200, (3,3), padding = 'same',activation = 'relu')(dense2)
dense3 = MaxPool2D(pool_size=2)(dense3)
dense3 = Dropout(0.2)(dense3)
dense4 = Conv2D(150, (3,3), padding = 'same',activation = 'relu')(dense3)
dense4 = Dropout(0.2)(dense4)
dense4 = Flatten()(dense4)
output1 = Dense(100,activation='softmax')(dense4)

model = Model(input = input1, output = output1)

# 3. 학습
model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics=['acc'])
modelpath = './model/exam/{epoch:02d}-{val_loss:.4f}.hdf5'    
earlystopping = EarlyStopping(monitor='loss', patience=5, mode = 'min')
modelcheckpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
hist = model.fit(x_train, y_train, batch_size=100, epochs=100, verbose=1, callbacks=[earlystopping, modelcheckpoint], validation_split=0.15,)


import matplotlib.pyplot as plt


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss: ', loss)
print('val_loss: ', val_loss)




plt.figure(figsize=(20,6))      # 10인치 * 6인치

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
print("loss : ", loss)
print("acc : ", acc)

print(y_predict)
print(y_test)
