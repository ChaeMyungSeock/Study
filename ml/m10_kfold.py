import pandas as pd
from sklearn.model_selection import train_test_split,KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import sklearn
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

# 1. 데이터
iris = pd.read_csv('./data/csv/iris.csv',header=0)

x = iris.iloc[:, 0:4] # 판다스에서 슬라이스 iloc 위치를 알고 있으면 됨  // loc 헤더와 인덱스 알고 있어야 함
y = iris.iloc[:,4]

# print(x)
# print(y)

kfold = KFold(n_splits=5, shuffle=True, random_state=666) 
# KfFold를 5개로 나눠서 수행하겠다. 20%의 데이터를 검증 80% 수행 => 5번 전체 데이터를 수행함으로

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 6667)



warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter = 'classifier')
# all_estimators => sklearn의 모든 classifier가 있음

# name, algorithm (변수명)으로 allAlgorithms이 반환값으로 반환함
#


# 교차 검증을 해주기 때문에 모든 train, validaton data, test data 모두를 k개의 데이터 셋을 만든 후
# k번 만큼 1) 학습 검증을 수행해 줌 


for (name, algroithm) in allAlgorithms:
    model = algroithm()
    
    scores = cross_val_score(model, x,y, cv=kfold)
    # fit의 역활을 수행 => 이 모델에 x,y의 데이터를 넣어서 cv=kfold를 수행해주겠다
    print(name, "의 정답률 = ", scores)
    # pred = cross_val_predict(scores, x,y)
    # print("pred : ", pred)
