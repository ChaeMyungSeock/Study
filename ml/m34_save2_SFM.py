'''
m28_eval2 와 3을 만들것

SelectFromModel 에
GridSearchcv
1. 회귀 rmse, mae   m29_eval1
2. 이진 분류 error  m29_eval2
3. 다중 분류 error  m29_eval3

1. eval에 'loss'와 다른 지표 1개 더 추가.
   29_1, 2, 3
2. earlyStopping 적용
3. plot으로 그릴것.

4. 결과는 주석으로 소스 하단에 표시 

5. m27 ~ 29까지 완벽 이해할것!

'''

# from m27_eval_metric.py

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, random_state=1)

model = XGBClassifier(n_estimators=1000, learning_rate=0.1)

# model.fit(x_train, y_train, verbose=True,  eval_metric= "error",
#                 eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=True,  eval_metric=["logloss","error"],
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)
# rmse, mae, logloss, error, auc

result = model.evals_result()
print(result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print(f"r2: {r2}")


thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit = True)

        parameter = {
            'n_estimators': [100, 200, 400],
            'learning_rate' : [0.03, 0.05, 0.07, 0.1],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'colsample_bylevel':[0.6, 0.7, 0.8, 0.9],
            'max_depth': [4, 5, 6]
        }

        search = GridSearchCV( XGBRegressor(), parameter, cv =5, n_jobs = -1)

        select_x_train = selection.transform(x_train)

        search.fit(select_x_train, y_train)

        select_x_test = selection.transform(x_test)
        x_pred = search.predict(select_x_test)

        score = r2_score(y_test, x_pred)
        print('R2는',score)

        print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
        model.save_model('./model/xgb_save/m34sfm_cancer/cancer.xgb' + str(select_x_train.shape[1])+'.model')

'''
