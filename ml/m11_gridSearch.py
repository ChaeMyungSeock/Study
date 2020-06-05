import pandas as pd
from sklearn.model_selection import train_test_split,KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.svm import SVC # 서포트 벡터 머신
from sklearn.model_selection import KFold, GridSearchCV


# 1. 데이터
iris = pd.read_csv('./data/csv/iris.csv',header=0)

x = iris.iloc[:, 0:4] # 판다스에서 슬라이스 iloc 위치를 알고 있으면 됨  // loc 헤더와 인덱스 알고 있어야 함
y = iris.iloc[:,4]

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 6667)


parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]
# 그리드 파라미터 사용.. 
# C : 1 kernel : lienear, C : 10 kernel : lienear, C : 100 kernel : lienear, C : 1000 kernel : lienear

kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(SVC(), parameters, cv=kfold)
# CV : cross validation
# 처음에 진짜 모델, 2번째 parameters, 3번째 kfold

# train test 8:2
# train 80% 교차검증

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
'''
최적의 매개변수 :  SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
 c가 1일 때 linear가 최상 기준 가장 높은 : acc 의 파라미터
  '''
print("최종 정답률 = ", accuracy_score(y_test, y_pred))

# kfold 먼저하고 train - test로 나눠도 됨 뭐가 더 좋을까