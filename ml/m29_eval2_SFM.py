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
model.fit(x_train, y_train, verbose=True,  eval_metric=["logloss","loss"],
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

'''
R2는 0.9058603832810583
Thresh=0.005, n=13, R2: 90.59%
R2는 0.9009087398931365
Thresh=0.008, n=12, R2: 90.09%
R2는 0.9093416908919445
Thresh=0.009, n=11, R2: 90.93%
R2는 0.9056896604113531
Thresh=0.009, n=10, R2: 90.57%
R2는 0.9148662676183015
Thresh=0.009, n=9, R2: 91.49%
R2는 0.9246197232545056
Thresh=0.015, n=8, R2: 92.46%
R2는 0.9293468003407601
Thresh=0.020, n=7, R2: 92.93%
R2는 0.9061280471137858
Thresh=0.021, n=6, R2: 90.61%
R2는 0.9147166499128062
Thresh=0.022, n=5, R2: 91.47%
R2는 0.8869769599185617
Thresh=0.032, n=4, R2: 88.70%
R2는 0.8776513104354586
Thresh=0.033, n=3, R2: 87.77%
R2는 0.7024282224661709
Thresh=0.160, n=2, R2: 70.24%
R2는 0.6659313577807507
Thresh=0.656, n=1, R2: 66.59%
'''