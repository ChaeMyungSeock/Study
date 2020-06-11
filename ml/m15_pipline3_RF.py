# RandomizedSearchCV + Pipline
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, 
                                                    shuffle = True, random_state = 666)



# makepipeline 쓸 때
parameters = [
    {'randomforestclassifier__n_estimators' : [1, 10, 100, 1000] },
    {'randomforestclassifier__n_estimators' : [1, 10, 100, 1000],  
                                   'randomforestclassifier__max_depth' : [2, 4, 6], 
                                   'randomforestclassifier__min_samples_leaf' : [2,3,4],
                                   'randomforestclassifier__min_samples_split' : [2,3,4]},
    {'randomforestclassifier__n_estimators' : [1, 10, 100, 1000], 
                                   'randomforestclassifier__max_depth' : [4,6,8],
                                   'randomforestclassifier__min_samples_split' : [3,4,5],
                                   }
]

# 그리드/랜덤 서치에서 사용할 매개 변수 그냥 pipline 쓸 때
# parameters = [{"svm__C" : [1, 10, 100, 1000], "svm__kernel":['linear']},
#               {"svm__C" : [1, 10, 100, 1000], "svm__kernel":['rbf'], "svm__gamma" :[0.001, 0.0001]},
#               {"svm__C" : [1, 10, 100, 1000], "svm__kernel":['sigmoid'], "svm__gamma" :[0.001, 0.0001]}] #20가지가 가능한 파라미터

# 그리드/랜덤 서치에서 사용할 매개 변수 그냥 pipline 쓸 때
# parameters = [{"svm__C" : [1, 10, 100, 1000], "svm__kernel":['linear']},
#               {"svm__C" : [1, 10, 100, 1000], "svm__kernel":['rbf'], "svm__gamma" :[0.001, 0.0001]},
#               {"svm__C" : [1, 10, 100, 1000], "svm__kernel":['sigmoid'], "svm__gamma" :[0.001, 0.0001]}] #20가지가 가능한 파라미터


# 2. 모델
# model = SVC()
# svc_model = SVC()

from sklearn.pipeline import  Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())]) # 모델을 모델명으로 사용시에 앞에 붙여줘야함 생각보다 네이밍이 중요한듯
pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())
model = RandomizedSearchCV(pipe,parameters, cv=5)

# pipeline, makepipline, parameter dict 확인 및 공부

# 3. 훈련
model.fit(x_train,y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc)

print("최적의 매개변수 : ", model.best_estimator_)

print(f"pipe.get_params():{pipe.get_params()}")
'''
pipe.get_params():{'memory': None, 'steps': [('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
 ('svm', SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False))], 'verbose': False, 'scaler': MinMaxScaler(copy=True, feature_range=(0, 1)), 'svm': SVC(C=1.0, break_ties=False, cache_size=200, 
    class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), 'scaler__copy': True, 'scaler__feature_range': (0, 1), 'svm__C': 1.0, 'svm__break_ties': False, 'svm__cache_size': 200, 
    'svm__class_weight': None, 'svm__coef0': 0.0, 'svm__decision_function_shape': 'ovr', 'svm__degree': 3, 'svm__gamma': 'scale', 'svm__kernel': 'rbf', 
    'svm__max_iter': -1, 'svm__probability': False, 'svm__random_state': None, 'svm__shrinking': True, 'svm__tol': 0.001, 'svm__verbose': False}
'''