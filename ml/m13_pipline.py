import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, 
                                                    shuffle = True, random_state = 666)


# 2. 모델
# model = SVC()
# svc_model = SVC()

from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm',SVC())])
# 전처리와 모델을 같이 돌림

pipe.fit(x_train, y_train)

print("acc : ",pipe.score(x_test,y_test))