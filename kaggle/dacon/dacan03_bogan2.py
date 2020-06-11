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
x_predict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0, index_col=0)

# print(train.head())
train = train.iloc[:,1:5]
x_predict = x_predict.iloc[:,1:5]


train = train.values
x_predict =x_predict.values
# print(submission.shape)
# print(x_prdeict.shape)
train = train.reshape(2800,375,4)
x_predict = x_predict.reshape(700,375,4)
exam_train = np.zeros((1,4),dtype=np.float64)

print(test.shape)
for i in range(700):

    # b = np.zeros((1,4),delattr)
    aaa = np.zeros((1,375,4),dtype=np.float64)
    aaa = x_predict[i,:,:]
    aaa = aaa.reshape(375,4)
    print(aaa.shape)
    print(len(aaa))
    size = 5            # time_steps = 5

    def split_x (dataset, size, y_column):
        x,y = list(), list()
        for i in range(len(dataset)):
            x_end = i + size
            y_end = x_end + y_column

            if y_end > len(dataset):
                break
            tem_x = dataset[i:x_end,:]
            tem_y = dataset[x_end:y_end,:]

            x.append(tem_x)
            y.append(tem_y)
        return np.array(x), np.array(y)

    x,y = split_x(aaa, 5, 1)
    y = y.reshape(y.shape[0],y.shape[1]*y.shape[2])

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1, shuffle=False)

    model = Sequential()
    model.add(LSTM(64, activation = leaky, input_shape = (5,4))) 
    model.add(Dense(100, activation = leaky))
    model.add(Dense(4, activation = leaky))

    earlystopping = EarlyStopping(monitor= 'loss', patience=5, mode = 'min')
    model.compile(optimizer='adam', loss = 'mse',metrics=['mse'])
    model.fit(x,y, epochs=20, batch_size=1, callbacks=[earlystopping],validation_split=0.1)


    loss, mse = model.evaluate(x_test, y_test)

    yhat = model.predict(x_test)
    # print(yhat)
    # print(yhat.shape)
    y_pred = yhat[-1,:]
    # print(y_pred.shape)
    y_pred = y_pred.reshape(1,4)
    exam_train = np.concatenate((exam_train,y_pred), axis=0)
print(exam_train)
np.save('./data/dacon/comp2/predict_split.npy',arr=exam_train)




# print(x.shape)
# print(y.shape)


# train1 = np.zeros((1,375,5),dtype=np.float64)
# def split_X(seq,size):
#     aaa = np.zeros((len(seq) - size+1,size,4),dtype=np.float64)
#     print(seq.shape)
#     idx = 0
#     for i in range(0,len(seq) - size +1): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
#         subset = seq[i: (i+size),:] # 한행에 넣을 데이터 추출
#         aaa[idx]= subset # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
#         idx += 1

#     print(aaa[0])
#     return (aaa, seq[len(seq)-size:] )


# train1 = split_X(train,50)








# def split_X(seq,size):
#     aaa = np.zeros((len(seq) - size+1,size,4),dtype=np.float64)
#     print(seq.shape)
#     idx = 0
#     for i in range(0,len(seq) - size +1): # len(seq) - size +1 = 몇개의 행을 갖을수 있는지 계산
#         subset = seq[i: (i+size),:] # 한행에 넣을 데이터 추출
#         aaa[idx]= subset # subset에 있는 item을 shape에 맞게 aaa 뒤에 행 추가
#         idx += 1

#     print(aaa[0])
#     return (aaa, seq[len(seq)-size:] )
'''
def split_xy10(dataset,size):
    aaa = np.zeros((len(dataset)- size +1 , size, 4),dtype = np.float64)
    print(dataset.shape)
    idx = 0
    for i in range(0,len(dataset) - size +1): # 몇 개의 행을 가질건지
        subset = dataset[i : (i+size),:]   # 한 행에 넣을 데이터 추출
        aaa[idx] = subset   # subset에있는 date를 shape에 맞게 aaa 뒤에 행 추가
        idx += 1

        print(aaa[0])
        return (aaa, dataset[len(dataset)-size : ])

train = split_xy10(train,2800)

#



# 2. 모델

input1 = Input(shape = (50,30))
dense1 = Conv1D(256, kernel_size=3,padding='same',activation='relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv1D(128, kernel_size=3,padding='same',activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv1D(64, kernel_size=3,padding='same',activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv1D(32, kernel_size=3,padding='same',activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(4, activation='relu')(dense1)

model = Model(input = input1, output = output1)


# 3. 훈련
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
earlystopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min')
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=[earlystopping])


# 4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)

y_predict = model.predict(x_prdeict)
print("loss : ", loss)
print("mse : ", mse)

# print(y_predict)
# print(y_test)
a = np.arange(2800,3500)

y_predict = pd.DataFrame(y_predict,a)
y_predict.to_csv('./data/dacon/comp2/sample_submission.csv', index = True, header=['X','Y','M','V'],index_label='id')
'''