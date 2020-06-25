import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRFRegressor,XGBClassifier,XGBRFClassifier,XGBRegressor
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score


train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape) # (10000,75) : x_train, test
print('test.shape :', test.shape)   # (10000,71) : x_predict
print('submmission.shape :', submission.shape)  # (10000,4)     : y_predict

train = train.interpolate() # 보간법 // 선형보간  // 데이터를 많이 짜를 경우에는 선형일 확률이 높음 따라서 선형보간을 사용 => 85점 정도
test = test.interpolate()


# print(train.head())
print(test.head())
# print(submission.head())


train = train.fillna(train.mean())
test = test.fillna(test.mean())

x_train = train.iloc[:, :71]
y_train = train.loc[:, 'hhb':'na']
print(x_train.shape)
print(y_train.shape)
x_train = x_train.values
y_train = y_train.values

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66)


model = XGBRegressor()
def XGboost_fit_acc(y_train, y_test):
    y_predict = []
    for i in range(len(submission.columns)):
        print(i)
        y_train1 = y_train[:, i]  
        model.fit(x_train, y_train1)
        
        y_test1 = y_test[:, i]
        score = model.score(x_test, y_test1)
        print('score: ', score)

        y_pred = model.predict(test)
        y_pred1 = model.predict(x_test)
        print('mae: ', mean_absolute_error(y_test, y_pred1))

        y_predict.append(y_pred)     
    return np.array(y_predict)

y_predict = XGboost_fit_acc(y_train, y_test).reshape(-1, 4) 


thresholds = np.sort(model.feature_importances_)

print(thresholds)


# for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     selection_x_train = selection.transform(x_train)
#     # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)

#     selection_model = XGBRegressor()
#     def XGboost_fit_acc(y_train, y_test):
#         y_predict = []
         

#         for i in range(len(submission.columns)):
#             print(i)
#             y_train1 = y_train[:, i]  
#             model.fit(x_train, y_train1)
        
#             y_test1 = y_test[:, i]
#             score = model.score(x_test, y_test1)
#             print('score: ', score)

#             y_pred = model.predict(test)
#             y_pred1 = model.predict(x_test)
#             print('mae: ', mean_absolute_error(y_test1, y_pred1))

#             y_predict.append(y_pred)
#         return np.array(y_predict)

#     selection_model.fit(selection_x_train, y_train)

#     select_x_test = selection.transform(x_test)
#     x_pred = selection_model.predict(select_x_test)

#     score = r2_score(y_test, x_pred)
#     print('R2는',score)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))



# x = train.iloc[:,:71]
# # print(x.head())
# print(x.shape)

# y = train.iloc[:,-4:]

# print(test.isnull())

# 서브밋파일 만든다.
# y_pred.to_csv(경로)


# x_train = x_train.reshape(x_train.shape[0],24)
# x_test = x_test.reshape(x_test.shape[0],24)
# my_imputer = Imputer
# x_train = my_imputer.fit_transform(x_train)
# y_train = my_imputer.fit_transform(y_train)

# 2. 모델


# def train_model(x_data, y_data, k=5):
#     models = []
    
#     k_fold = KFold(n_splits=k, shuffle=True, random_state=85)
    
#     for train_idx, val_idx in k_fold.split(x_data):
#         x_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
#         x_val, y_val = x_data.iloc[val_idx], y_data[val_idx]
    
#         d_train = xgb.DMatrix(data = x_train, label = y_train)
#         d_val = xgb.DMatrix(data = x_val, label = y_val)
        
#         wlist = [(d_train, 'train'), (d_val, 'eval')]
        
#         params = {
#             'objective': 'reg:squarederror',
#             'eval_metric': 'mae',
#             'seed':777
#             }

#         model = xgb.train(params=params, dtrain=d_train, num_boost_round=500, verbose_eval=500, evals=wlist)
#         models.append(model)
    
#     return models
# models = {}
# for label in y_train.columns:
#     print('train column : ', label)
#     models[label] = train_model(x_train, y_train[label])

# for col in models:
#     preds = []
#     for model in models[col]:
#         preds.append(model.predict(xgb.DMatrix(test.loc[:,:])))
#     pred = np.mean(preds, axis=0)

#     submission[col] = pred
   

# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(submission,a)
# # y_pred.to_csv('./data/dacon/comp1/sub_XGB2.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')




# RandomForestRegressor
# mae :  1.5219343999999997

# GradientBoostingRegressor
# mae : 3.4783443467	