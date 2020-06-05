# Randomforest 적용
# cifar10 적용

import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split,KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.svm import SVC # 서포트 벡터 머신
from sklearn.model_selection import KFold, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 6667)
# x_train = x_train.values
# y_train = y_train.values

print(x_train.__class__)

parameters = {"n_estimators":[10,100,1000], 'max_features' : [2,4,6], 'max_depth' : [4,4,4]}
    # {'bootstrap':[False],"n_estimators":[3,10], 'max_features':[2,3,4]}

'''
n_estimator : 결정 트리의 개수, default 값은 10, 많을 수록 좋은 성능이 나올 수 도 있지만, 무조건적인것은 아님
max_features : 데이터의 feature를 참조할 비율, 개수를 뜻함. default는 auto
max_depth : 트리의 깊이를 뜻합니다.
min_samples_leaf : 리프노드가 되기 위한 최소한의 샘플 데이터 수
min_samples_split : 노드를 분할하기 위한 최소한의 데이터 수
'''
# C : 1 kernel : lienear, C : 10 kernel : lienear, C : 100 kernel : lienear, C : 1000 kernel : lienear

kfold = KFold(n_splits=2, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_jobs=-1)
# n_jobs 여러 cpucore지원
# CV : cross validation
# 처음에 진짜 모델, 2번째 parameters, 3번째 kfold

# train test 8:2
# train 80% 교차검증

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)

# print("가장 중요한 파라미터 : ", model.feature_importances_)

print("최종 정답률 = ", accuracy_score(y_test, y_pred))

'''
최적의 매개변수 :  SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
 c가 1일 때 linear가 최상 기준 가장 높은 : acc 의 파라미터
  
print("최종 정답률 = ", accuracy_score(y_test, y_pred))

# kfold 먼저하고 train - test로 나눠도 됨 뭐가 더 좋을까

'''