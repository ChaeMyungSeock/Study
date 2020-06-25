import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D,Conv1D,MaxPool1D,Dropout
from keras.layers import Dropout
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


train_features = pd.read_csv('./data/dacon/comp2/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv')
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv')
test_features = pd.read_csv('./data/dacon/comp2/test_features.csv')


# print(train.head())
# print(test.head())


# x_data = train.iloc[:, 1:]                           
# y_data = test.iloc[:,:]
# x_prdeict = x_prdeict.iloc[:,1:]

# train_target = train_target.iloc[:,3:5]
print(test_features.head())


fs = 5
# sampling frequency 
fmax = 25
# sampling period
dt = 1/fs
# length of signal
N  = 75

df = fmax/N
f = np.arange(0,N)*df

xf = np.fft.fft(train_features[train_features.id==10]['S1'].values)*dt
print(len(np.abs(xf[0:int(N/2+1)])))
# Fourier transformation and the convolution theorem
def autocorr1(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    return r2[:len(x)//2]

train_ids = train_features.drop_duplicates(['id'])['id'].values

from tqdm import tqdm

signals = []

for i in tqdm(train_ids):
    xf1 = np.fft.fft(train_features[train_features.id==i]['S1'].values)*dt
    xf2 = np.fft.fft(train_features[train_features.id==i]['S2'].values)*dt
    xf3 = np.fft.fft(train_features[train_features.id==i]['S3'].values)*dt
    xf4 = np.fft.fft(train_features[train_features.id==i]['S4'].values)*dt
    
    signals.append(np.concatenate([np.abs(xf1[0:int(N/2+1)]), np.abs(xf2[0:int(N/2+1)]), np.abs(xf3[0:int(N/2+1)]), np.abs(xf4[0:int(N/2+1)])]))
    
signals = np.array(signals)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
)
model.fit(signals, train_target)

test_ids = test_features.drop_duplicates(['id'])['id'].values


test_signals = []

for i in tqdm(test_ids):
    xf1 = np.fft.fft(test_features[test_features.id==i]['S1'].values)*dt
    xf2 = np.fft.fft(test_features[test_features.id==i]['S2'].values)*dt
    xf3 = np.fft.fft(test_features[test_features.id==i]['S3'].values)*dt
    xf4 = np.fft.fft(test_features[test_features.id==i]['S4'].values)*dt
    
    test_signals.append(np.concatenate([np.abs(xf1[0:int(N/2+1)]), np.abs(xf2[0:int(N/2+1)]), np.abs(xf3[0:int(N/2+1)]), np.abs(xf4[0:int(N/2+1)])]))
    
test_signals = np.array(test_signals)
y_pred = model.predict(test_signals)

print(y_pred)
submission = submission.iloc[:,:5]
print(submission.head())

for i in range(4):
    submission.iloc[:,i] = y_pred[:,i]

submission =submission.values
a = np.arange(2800,3500)
#np.arange--수열 만들때
submission = pd.DataFrame(submission, a)

#np.arange--수열 만들때
submission = submission.iloc[:,1:5]
print(submission.head())

submission.to_csv("./data/dacon/comp2/sample_submission2_2.csv", header = ["X","Y","M","V"], index = True, index_label="id" )
# from sklearn.metrics import mean_squared_error
# mse=mean_squared_error(,y_pred)
# print("mse:",mse)

# submit = pd.read_csv('sample_submission.csv')

# submit.head()


'''
x_data = x_data.values
y_data = y_data.values
x_prdeict = x_prdeict.values

# print('train.shape :', train.shape) # (10500000,75) : x_train, test
# print('test.shape :', test.shape)   # (2800,4) : x_predict
# print('x_predict.shape :', x_prdeict.shape) # (262500,75) : x_train, test
# print('submmission.shape :', submission.shape)  # (700,4)     : y_predict

x_data = x_data.reshape(2800,375,4)
x_prdeict = x_prdeict.reshape(700,375,4)


x_data = x_data.reshape(2800,375*4)
x_prdeict = x_prdeict.reshape(700,375*4)

print(y_data.shape)
print(y_data.__class__)



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state =33)

print(y_test.shape)
print(y_train.shape)

# model = DecisionTreeRegressor(max_depth =4)                     # max_depth 몇 이상 올라가면 구분 잘 못함
# model = RandomForestRegressor(n_estimators = 200, max_depth=3)
# model = GradientBoostingRegressor()

model = xgb.XGBRFRegressor(eta=0.1, max_depth = 5, colsample_bytree = 0.5)
model.fit(x_train,y_train)
y_testpred = model.predict(x_test)
# y_test = pca.inverse_transform(y_test)
# y_testpred = pca.inverse_transform(y_testpred)
score = model.score(x_test,y_test)
print(score)
y4 = model.predict(x_prdeict)
print(y4.shape)
y4=y4.reshape(y4.shape[0],1)
y4 = pca.inverse_transform(y4)
y4 = scaler.inverse_transform(y4)
print(y_testpred.shape)
print(y4.shape)




print(y4.shape)
print(y4)
print('mse: ', mean_squared_error(y_test, y_testpred))


a = np.arange(2800,3500)



y_predict = pd.DataFrame(y4,a)
y_predict.to_csv('./data/dacon/comp2/com2_result4.csv', index = True, header=['X','Y','M','V'], index_label='id')


'''