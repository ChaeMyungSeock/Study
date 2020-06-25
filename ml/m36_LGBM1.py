import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor
import pickle

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, random_state=1)

# train_ds = lgb.Dataset(x_train, label = y_train)
# test_ds = lgb.Dataset(x_test, label = y_test)

# params = {'learning_rate' : 0.01, 'max_depth': 13 , 'boosting' : 'gbdt',
#         'objective':'regression', 'metric':'mse', 'is_training_metric':True,
#         'num_leaves':144, 'feature_fraction':0.9, 'bagging_fraction':0.7,
#         'bagging_freq':5, 'seed':2020}

# model = lgb.train(params, train_ds, 1000, test_ds,verbose_eval=100, early_stopping_rounds=100)

parameter = {
            'num_iterations': 1000,           
            'learning_rate' : 0.05,
            'early_stopping_round': 20,
            
        }
model = LGBMRegressor()

model.fit(x_train, y_train, verbose=True,  eval_metric= "error",
                eval_set=[(x_train, y_train), (x_test, y_test)],early_stopping_rounds=100)

# rmse, mae, logloss, error, auc


y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print(f"r2: {r2}")


thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit = True)

        search = LGBMRegressor()

        select_x_train = selection.transform(x_train)

        search.fit(select_x_train, y_train)

        select_x_test = selection.transform(x_test)
        x_pred = search.predict(select_x_test)

        score = r2_score(y_test, x_pred)
        # print('R2는',score)

        print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
        # model.save_model('./model/xgb_save/m34sfm/cancer.xgb' + str(thresh)+'.model')
        pickle.dump(model, open('./model/lgb_save/m36_lgbm1/m36.lgb' + str(thresh)+'.dat', 'wb'))

'''
[0.00497141 0.00802845 0.00874821 0.00903318 0.00930241 0.01546535
 0.02015997 0.02073635 0.02245208 0.03186943 0.03302769 0.15981925
 0.65638626]
Thresh=0.005, n=13, R2: 90.59%
Thresh=0.008, n=12, R2: 90.09%
Thresh=0.009, n=11, R2: 90.93%
Thresh=0.009, n=10, R2: 90.57%
Thresh=0.009, n=9, R2: 91.49%
Thresh=0.015, n=8, R2: 92.46%
Thresh=0.020, n=7, R2: 92.93%
Thresh=0.021, n=6, R2: 90.61%
Thresh=0.032, n=4, R2: 88.70%
Thresh=0.033, n=3, R2: 87.77%
Thresh=0.160, n=2, R2: 70.24%
Thresh=0.656, n=1, R2: 66.59%
'''