import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA


train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape) # (10000,75) : x_train, test
print('test.shape :', test.shape)   # (10000,71) : x_predict
print('submmission.shape :', submission.shape)  # (10000,4)     : y_predict
# print(train.head())
# y_train = train.loc[:,'hhb':'na']

# print(y_train.head())
train = train.interpolate() # 보간법 // 선형보간  // 데이터를 많이 짜를 경우에는 선형일 확률이 높음 따라서 선형보간을 사용 => 85점 정도
test = test.interpolate()



# print(len(train.index))
# print(train.iloc[:,1])
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# for i in train.columns:
#     # print(i)
#     print(len(train[train[i].isnull()]))

# for i in test.columns:
#     # print(i)
#     print(len(test[test[i].isnull()]))
# # print(train.isnull().sum())

x = train.iloc[:,:71]
# print(x.head())
print(x.shape)

y = train.loc[:,'hhb':'na']

# print(test.isnull())

# 서브밋파일 만든다.
# y_pred.to_csv(경로)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle = True,random_state = 666)

print(x_train.shape)
print(y_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

pca = PCA(n_components=24)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
test = pca.transform(test)

x_train = x_train.reshape(x_train.shape[0],24)
x_test = x_test.reshape(x_test.shape[0],24)

# 2. 모델

input1 = Input(shape=(24,))
x = Dense(64, activation='relu')(input1)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)
output1 = Dense(4,activation='relu')(x)

model = Model(inputs = input1, outputs = output1)



# 3. 학습
model.compile(optimizer = 'adam', loss= 'mae', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=30, verbose=1, validation_split=0.2)


# 4. 평가 예측

loss, mae = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", mae)

# print(y_predict)
# print(y_test)



y_pred = model.predict(test)
print(y_pred.shape)
a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# print(y_pred)
