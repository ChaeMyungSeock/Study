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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.preprocessing import sequence

# LSTM형식인데 시계열인지 명확하지 않으면 CNN으로 해야함 시계열 형식이 375단위로 잘리기 때문
# why? 잘라서 특징을 추출하기 때문

train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header=0, index_col=0)
x_prdeict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0, index_col=0)

print(train.head())

train = train.values
x_prdeict =x_prdeict.values
test = test.values
print(submission.shape)
print(x_prdeict.shape)
train = train.reshape(2800,375,5)
x_prdeict = x_prdeict.reshape(700,375,5)
y_pred = []
y_predict = []

# mx = 0
# def Max_fuction(fu):
#     if (fu>= mx):
#         mx = fu
#     return mx

train_time = train[0,:,:]

# print(train[0,0,4])
# print(train_time.shape)
train1 = np.zeros((1,5),dtype=np.float64)
print(train1)
train2 = list()

# for i in range(2800):
for j in range(375):
    for k in range(4):
        if(train[0,j,k+1] != 0):
            first = j
            train1 = np.append(train1,first)
            if((train[0,j,1] !=0 ) and (train[0,j,2]!=0 )and (train[0,j,3]!=0) and (train[0,j,4]!=0)):
                train2.append(j)
                
        break 
            
    # train1 = train1.reshape(5,int(train1.shape[0]/15))
print(train1)
print(train2)

    # a = int(train1[5])
    # b = train2[0]
    # print(a )
    # print(b )

    # Max = a-b



'''
    # print(train[0,a:b+1,:])
    x= train[0,a:b+1,:]
    print(x.shape)
    y = test[0,:]

    print(y.shape)
    x_train, y_train = train_test_split(x,y, test_size = 0, shuffle=True, random_state = 33)

    model = Sequential()
    model.add(Dense(128, input_shape=(5,), activation=leaky))
    model.add(Dense(64, activation=leaky))
    model.add(Dense(32, activation=leaky))
    model.add(Dense(5, activation=leaky))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(x,y,batch_size=1,epochs=10, validation_split=0.2)


for q in range(700):
    x_a_predict = x_prdeict[q,a:b,:]
    y_predict = model.predict(x_a_predict)
    y_pred.append(y_predict)






# print(train1)
# train1 = train1.reshape(5,int(train1.shape[0]/5))
# print(train1.shape)
            



# for i in range(2800):
# aaa = np.zeros((1,375,5),dtype=np.float64)
# aaa = train[1,:,:]
# print(aaa.shape)



# train_timestep1 = train[0,:,1]
# train_timestep2 = train[0,:,2]
# train_timestep3 = train[0,:,3]
# train_timestep4 = train[0,:,4]

# train_time = train_time.reshape(train_time.shape[0],1)
# train_timestep1 = train_timestep1.reshape(train_timestep1.shape[0],1)
# train_timestep2 = train_timestep2.reshape(train_timestep2.shape[0],1)
# train_timestep3 = train_timestep3.reshape(train_timestep3.shape[0],1)
# train_timestep4 = train_timestep4.reshape(train_timestep4.shape[0],1)






# print(train1.shape)

print(submission.shape)
print(x_prdeict.shape)

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