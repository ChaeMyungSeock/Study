import numpy as np
from keras.models import Sequential,Input,Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten,Dropout
from keras.callbacks import EarlyStopping


target_train1 = np.loadtxt('./Hexapod_Bot/image/data/y_train_1.txt',dtype=int)
target_train0 = np.loadtxt('./Hexapod_Bot/image/data/y_train_0.txt',dtype=int)
target_test1 = np.loadtxt('./Hexapod_Bot/image/data/y_test_1.txt',dtype=int)
target_test0 = np.loadtxt('./Hexapod_Bot/image/data/y_test_1.txt',dtype=int)



train_data0 = np.load('./Hexapod_Bot/image/data/use_data/train/train_0.npy')
train_data1 = np.load('./Hexapod_Bot/image/data/use_data/train/train_1.npy')

test_data0 = np.load('./Hexapod_Bot/image/data/use_data/test/test_0.npy')
test_data1 = np.load('./Hexapod_Bot/image/data/use_data/test/test_1.npy')

# print(train_data0.shape)
# print(train_data1.shape)
# print(test_data0.shape)
# print(test_data1.shape)

train_data0 = train_data0.reshape(train_data0.shape[0],train_data0.shape[1]*train_data0.shape[2]*train_data0.shape[3])
train_data1 = train_data1.reshape(train_data1.shape[0],train_data1.shape[1]*train_data1.shape[2]*train_data1.shape[3])
test_data0 = test_data0.reshape(test_data0.shape[0],test_data0.shape[1]*test_data0.shape[2]*test_data0.shape[3])
test_data1 = test_data1.reshape(test_data1.shape[0],test_data1.shape[1]*test_data1.shape[2]*test_data1.shape[3])

# print(train_data0.shape)
# print(train_data1.shape)
# print(test_data0.shape)
# print(test_data1.shape)


x_data = np.vstack((train_data0,train_data1))
x_train = x_data.reshape(x_data.shape[0],360,360,3)

x_data1 = np.vstack((test_data0,test_data1))
x_test = x_data1.reshape(x_data1.shape[0],360,360,3)


print(target_test0.shape)
print(target_test1.shape)
print(target_train0.shape)
print(target_train1.shape)
y_train = np.hstack((target_train0, target_train1))
y_test = np.hstack((target_test0,target_test1))
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 2. 모델
# input1 = Input(shape = (360,360,3))
# dense1 = Conv2D(32, (3,3), padding = 'valid')(input1)
# dense2 = Conv2D(32, (3,3), padding = 'valid')(dense1)
# dense3 = Dropout(0.3)(dense2)
# dense4 = Conv2D(32, (3,3), padding = 'valid')(dense3)
# dense5 = Dropout(0.3)(dense4)
# dense6 = Conv2D(32, (3,3), padding = 'valid')(dense5)
# dense7 = MaxPool2D(pool_size = 2)(dense6)
# dense8 = Dropout(0.3)(dense7)
# dense9 = Flatten()(dense8)
# output1 = Dense(1,activation = 'sigmoid')(dense9)

# model = Model(input = input1, output = output1)


# # 3. 훈련
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
# earlystopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')
# model.fit(x_train, y_train, batch_size=100, epochs=2, validation_split=0.1, callbacks=[earlystopping])


# # 4. 평가, 예측

# loss, acc = model.evaluate(x_train, y_train)

# y_predict = model.predict(x_test)
# print("loss : ", loss)
# print("acc : ", acc)

# print(y_predict)
# print(y_test)
