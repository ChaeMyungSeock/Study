import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# %matplotlib inline

# quailty y값 =>1 x=> 11칼럼
raw_data = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=';', encoding='cp949')
raw_data.info()

print(raw_data.head())

# print(raw_data.isnull())

y = raw_data["quality"]

# print(y.__class__)
y = y.values
raw_data = raw_data.values

# print(raw_data.shape)

x = raw_data[:,:11]
print(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 80)


scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model = RandomForestRegressor() 
model = RandomForestClassifier(n_estimators=100,min_samples_split=2,max_depth=18,random_state=30)

model.fit(x_train, y_train)

# 4. 평가 예측

y_predict = model.predict(x_test)
score = model.score(x_test,y_test)


# acc = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
# accuracy_score => evaluate와 비슷

print("와인의 등급 : ", y_predict)
print("mse : ", mse)
print("r2 : ", r2)
#
print("score : ", score)
acc = accuracy_score(y_test, y_predict)
print("acc : ", acc)

# mse :  0.34285714285714286
# r2 :  0.5811028109670459
# score :  0.7408163265306122
# acc :  0.7408163265306122