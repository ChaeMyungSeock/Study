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


train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0,index_col=0)
test = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header=0)
x_prdeict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0,index_col=0)

# print(test.dtypes)
# print(train.head())
# print(test.head())
# print(x_prdeict.head())
# print(submission.head())

# print('train.shape :', train.shape) # (10500000,75) : x_train, test
# print('test.shape :', test.shape)   # (2800,4) : y_train
# print('x_predict.shape :', x_prdeict.shape) # (262500,75) : x_train, test
# print('submmission.shape :', submission.shape)  # (700,4)     : y_predict

# print(train.head())
x_data = train.iloc[:, 1:]                           
y_data = test.iloc[:,:]
x_prdeict = x_prdeict.iloc[:,1:]
# print(test.head())

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


scaler = StandardScaler()
scaler.fit(y_data)
y_data = scaler.transform(y_data)
pca = PCA(n_components=1)
pca.fit(y_data)
y_data = pca.transform(y_data)
print(y_data.shape)


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



# def tree_fit(y_train, y_test):
#     model.fit(x_train, y_train)
#     score = model.score(x_test, y_test)
#     print('score: ', score)
#     y_predict = model.predict(x_prdeict)
#     y_pred1 = model.predict(x_test)
#     print('mae: ', mean_absolute_error(y_test, y_pred1))
#     return y_predict


# def boost_fit_acc(y_train, y_test):
#     y_predict = []
#     for i in range(len(submission.columns)):
#        print(i)
#        y_train1 = y_train[:, i]  
#        model.fit(x_train, y_train1)
       
#        y_test1 = y_test[:, i]
#        score = model.score(x_test, y_test1)
#        print('score: ', score)

#        y_pred = model.predict(x_prdeict)
#        y_pred1 = model.predict(x_test)
#        print('mae: ', mean_absolute_error(y_test1, y_pred1))

#        y_predict.append(y_pred)
#     return np.array(y_predict)

# y_predict = tree_fit(y_train, y_test)
# y_predict = boost_fit_acc(y_train, y_test)

print(y4.shape)
print(y4)
print('mse: ', mean_squared_error(y_test, y_testpred))


a = np.arange(2800,3500)



y_predict = pd.DataFrame(y4,a)
y_predict.to_csv('./data/dacon/comp2/com2_result4.csv', index = True, header=['X','Y','M','V'], index_label='id')

'''
print(model.feature_importances_)


## feature_importances
def plot_feature_importances(model):
    plt.figure(figsize= (10, 40))
    n_features = x_data.shape[1]                                # n_features = column개수 
    plt.barh(np.arange(n_features), model.feature_importances_,      # barh : 가로방향 bar chart
              align = 'center')                                      # align : 정렬 / 'edge' : x축 label이 막대 왼쪽 가장자리에 위치
    plt.yticks(np.arange(n_features), x_data.columns)          # tick = 축상의 위치표시 지점
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)             # y축의 최솟값, 최댓값을 지정/ x는 xlim

plot_feature_importances(model)
plt.show()


a = np.arange(2800,3500)

y_predict = pd.DataFrame(y4,a)
y_predict.to_csv('./data/dacon/comp2/com2_result3.csv', index = True, header=['X','Y','M','V'], index_label='id')
'''