from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_boston()

# x = dataset.data
# y = dataset.target


x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.__class__)
print(x_train.shape)

print(y_train.__class__)
print(y_train.shape)


model = XGBRegressor(n_estimators=1000)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 : ",score)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

import time
start = time.time()
for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)

    selection_model = XGBRegressor(n_estimators=2000)
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))

end = time.time() - start

start2 = time.time()
for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)

    selection_model = XGBRegressor(n_jobs= -1 ,n_estimators=2000)
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))

end2 = time.time() - start2
print('걸린시간 : ',end)
print('n_jobs 걸린시간 : ',end2)
