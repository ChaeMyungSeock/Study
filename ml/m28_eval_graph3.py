# from m27_eval_metric.py

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
dataset = load_iris()

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, random_state=1)

model = XGBClassifier(n_estimators=1000, learning_rate=0.1,n_jobs=-1, objective='multi:softmax')
# model = MultiOutputClassifier(xgb)
# model.fit(x_train, y_train, verbose=True,  eval_metric= "error",
#                 eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=True,  eval_metric=["mlogloss","merror"],
            eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=20)


# rmse, mae, logloss, error, auc

result = model.evals_result()
print(result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print(f"r2: {r2}")

epochs = len(result['validation_0']['mlogloss'])
x_axis = range(0, epochs)

## 그래프, 시각화
epochs = len(result['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mlogloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['mlogloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['merror'], label = 'Train')
ax.plot(x_axis, result['validation_1']['merror'], label = 'Test')
ax.legend()
plt.ylabel('Error')
plt.title('XGBoost Error')
plt.show()
'''
r2: 0.9427480916030534
'''