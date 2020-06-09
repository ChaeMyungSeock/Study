import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D,Conv1D,MaxPool1D,Dropout
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)
# LSTM형식인데 시계열인지 명확하지 않으면 CNN으로 해야함 시계열 형식이 375단위로 잘리기 때문
# why? 잘라서 특징을 추출하기 때문

train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header=0, index_col=0)
x_prdeict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0, index_col=0)

# print(train.head())
# print(test.head())
# print(x_prdeict.head())
# print(submission.head())

print('train.shape :', train.shape) # (10500000,75) : x_train, test
print('test.shape :', test.shape)   # (2800,4) : x_predict
print('x_predict.shape :', x_prdeict.shape) # (262500,75) : x_train, test
print('submmission.shape :', submission.shape)  # (700,4)     : y_predict
train = train.values
x_prdeict = x_prdeict.values

train = train.reshape(2800,25,75)
x_prdeict = x_prdeict.reshape(700,25,75)

x_train, x_test,y_train,y_test = train_test_split(train,test, test_size=0.1,shuffle = True, random_state = 66)

x_train = x_train.reshape(x_train.shape[0],25*75)
x_test = x_test.reshape(x_test.shape[0],25*75)
x_prdeict = x_prdeict.reshape(700,25*75)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_prdeict = scaler.transform(x_prdeict)

print(x_train.shape)
# train = train.reshape(2800,25,75)
# x_prdeict = x_prdeict.reshape(700,25,75)

pca = PCA(n_components=180)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
x_prdeict = pca.transform(x_prdeict)


x_train = x_train.reshape(x_train.shape[0],10,18)
x_test = x_test.reshape(x_test.shape[0],10,18)
x_prdeict = x_prdeict.reshape(700,10,18)

# 2. 모델

input1 = Input(shape = (10,18))
dense1 = Conv1D(512, kernel_size=5,padding='same',activation=leaky)(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv1D(256, kernel_size=5,padding='same',activation=leaky)(dense1)
dense1 = Dropout(0.5)(dense1)
dense1 = Conv1D(128, kernel_size=5,padding='same',activation=leaky)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv1D(64, kernel_size=5,padding='same',activation=leaky)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(4, activation=leaky)(dense1)

model = Model(input = input1, output = output1)


# 3. 훈련
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
earlystopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min')
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.1, callbacks=[earlystopping])


# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)

y_predict = model.predict(x_prdeict)
print("loss : ", loss)
print("mse : ", mse)

# print(y_predict)
# print(y_test)
a = np.arange(2800,3500)

y_predict = pd.DataFrame(y_predict,a)
y_predict.to_csv('./data/dacon/comp2/sample_submission.csv', index = False)
