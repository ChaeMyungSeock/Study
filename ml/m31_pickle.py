# xgboost evaluate

import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer


## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBClassifier(n_estimators = 1000,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = False, eval_metric = 'error',
          eval_set = [(x_train, y_train), (x_test, y_test)])
# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

# results = model.evals_result()
# print("eval's result : ", results)


y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print('acc : ', acc)

import pickle   # 파이썬에서 제공
pickle.dump(model, open('./model/xgb_save/cancer.pickle.dat', 'wb' ))

'''
(open(‘text.txt’, ‘w’) 방식으로 데이터를 입력하면 string 자료형으로 저장된다.)
pickle로 데이터를 저장하거나 불러올때는 파일을 바이트형식으로 읽거나 써야한다. (wb, rb)
wb로 데이터를 입력하는 경우는 .bin 확장자를 사용하는게 좋다.
모든 파이썬 데이터 객체를 저장하고 읽을 수 있다.
'''

print('저장됐다.')

model2 = pickle.load( open('./model/xgb_save/cancer.pickle.dat', 'rb' ))
print('불러왔다.')

y_pred = model.predict(x_test)


acc = accuracy_score(y_pred, y_test)
print('acc : ', acc)
