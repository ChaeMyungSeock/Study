import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMRegressor


train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

x = np.load('./dacon/comp1/x_train.npy')
y = np.load('./dacon/comp1/y_train.npy')
x_predict = np.load('./dacon/comp1/x_pred.npy')


print(x.shape)
print(y.shape)
print(x_predict.shape)

x = pd.DataFrame(x)
x_predict = pd.DataFrame(x_predict)
clf = IsolationForest(max_samples=1000, random_state=1)
clf.fit(x)
pred_outliers = clf.predict(x)
out = pd.DataFrame(pred_outliers)
out = out.rename(columns={0:"out"})
new_train = pd.concat([x, out],1)

clf = IsolationForest(max_samples=1000, random_state=1)
clf.fit(x_predict)
pred_outliers = clf.predict(x_predict)
out = pd.DataFrame(pred_outliers)
out = out.rename(columns={0:"out"})
new_test = pd.concat([x_predict, out],1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# 2. 모델

#2. feature_importance
xgb = LGBMRegressor()
multi_XGB = MultiOutputRegressor(xgb)
multi_XGB.fit(x_train, y_train)

# print(len(multi_XGB.estimators_))   # 4


# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)



for i in range(len(multi_XGB.estimators_)):
    threshold = np.sort(multi_XGB.estimators_[i].feature_importances_)

    for thres in threshold:
        selection = SelectFromModel(multi_XGB.estimators_[i], threshold = thres, prefit = True)
        
        parameter = {
            'n_estimators': [1000, 2000, 4000],
            'learning_rate' : [0.05, 0.07, 0.1]
        }
        search = GridSearchCV(LGBMRegressor() , parameter, cv =5, n_jobs=-1)
        

        select_x_train = selection.transform(x_train)

        multi_search = MultiOutputRegressor(search)
        multi_search.fit(select_x_train, y_train )
        
        select_x_test = selection.transform(x_test)

        y_pred = multi_search.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
        # print(search.best_params_)
 
        select_x_pred = selection.transform(x_predict)
        y_predict = multi_search.predict(select_x_pred)
        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./dacon/comp1/sub_msbsoot%i_%.5f.csv'%(select_x_train.shape[1], mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
        print("mae : ", mae)
