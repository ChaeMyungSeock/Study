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
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error


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

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,shuffle = True,random_state = 87)

print(x_train.shape)
print(y_train.shape)



# x_train = x_train.reshape(x_train.shape[0],24)
# x_test = x_test.reshape(x_test.shape[0],24)

# 2. 모델



pipe = Pipeline([("scaler",StandardScaler()),('model',DecisionTreeRegressor())]) 

pipe.fit(x_train , y_train)


y_predict=pipe.predict(x_test)

from sklearn.metrics import mean_squared_error
# mse=mean_squared_error(y_test,y_predict)
# print("mse:",mse)
mae = mean_absolute_error(y_test,y_predict)
print('mae : ',mae)
result=pipe.predict(test)
print(y_predict.shape)
print(y_predict)



# print(y_predict)
# print(y_test)

# print(y_pred.shape)
a = np.arange(10000,20000)
y_pred = pd.DataFrame(result,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# print(y_pred)

# DecisionTreeRegressor
# mae :  2.1713449999999996