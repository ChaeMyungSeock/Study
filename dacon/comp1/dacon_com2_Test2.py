import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D,Conv1D,MaxPool1D,Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
# LSTM형식인데 시계열인지 명확하지 않으면 CNN으로 해야함 시계열 형식이 375단위로 잘리기 때문
# why? 잘라서 특징을 추출하기 때문
from xgboost import XGBClassifier,XGBModel
from xgboost.sklearn import XGBRFRegressor,XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

dataset_train = np.load('./data/dacon/comp2/sample_train_data.npy')
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0, index_col=0)
dataset_test = np.load('./data/dacon/comp2/sample_test_data.npy')
x_prdeict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0, index_col=0)
x_prdeict = x_prdeict.values
print(x_prdeict.shape)
x_prdeict = x_prdeict.reshape(700,375,5)
train1 = np.zeros((1,5),dtype=np.float64)
test = []
for i in range(700):
    test = x_prdeict[i,23:26,:]
    train1 = np.append(train1, test)


train1 = train1.reshape(2101,5)
train1 = train1[1:,:]
# print(train1.shape)
# print(train1)

train1 = train1.reshape(700,3,5)


print(dataset_test.shape)
print(train1.shape)
r_test = np.zeros((1,5),dtype=np.float64)
real = np.zeros((1,5),dtype=np.float64)

for i in range(700):
   r_test = np.vstack([dataset_test[i,:,:], train1[i,:,:]])
   real = np.append(real, r_test)

print(real.shape)
real = real.reshape(13301,5)
real = real[1:,:]
real = real.reshape(700,19,5)
print(real.shape)

np.save('./data/dacon/comp2/predict_data.npy',real)
# r_test = r_test.reshape(700,19,5)

# print(train1)
# dtaaset 23부터 25까지

print(dataset_train.shape)
print(dataset_test.shape)
# print(dataset_train)
# print(dataset_test)
train_target = train_target.iloc[:,0:2]
# print(train_target.head())
train_target = train_target.values

# print(dataset_test.shape)
# print(dataset_train.shape)


x_train,x_test,y_train,y_test=train_test_split(dataset_train,train_target,test_size=0.2)

# dataset_train = dataset_train.reshape(dataset_train.shape[0], dataset_train.shape[1]*dataset_train.shape[2])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

dataset_test=dataset_test.reshape(700,16*5) 

# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)
# pca = PCA(n_components=80)
# x_train = pca.fit_transform(x_train)
# x_test= pca.fit_transform(x_test)

# scaler.fit(dataset_test)
# dataset_test = scaler.transform(dataset_test)


print(x_train.shape) # (2240,19,4)
print(x_test.shape)  # (560,19,4)
# print(dataset_test.shape) # (700, 16, 5)
# x_train=x_train.reshape(2240,19*5) # (2240, 80)
# x_test=x_test.reshape(560,19*5) # (700, 80)



model = RandomForestRegressor(n_estimators=1000,max_depth=3)
model.fit(x_train , y_train)


y_predict=model.predict(x_test)

result=model.predict(dataset_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_predict)
print("mse:",mse)


print(y_predict.shape)
print(y_predict)




# a = np.arange(2800,3500)
# #np.arange--수열 만들때
# submission = result
# submission = pd.DataFrame(submission, a)
# submission.to_csv("./data/dacon/comp3/sample_submission1_2.csv", header = ["X","Y","M","V"], index = True, index_label="id" )
