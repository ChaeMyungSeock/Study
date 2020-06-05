import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import sklearn
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv',header=1)

x = boston.iloc[:, 0:13] # 판다스에서 슬라이스 iloc 위치를 알고 있으면 됨  // loc 헤더와 인덱스 알고 있어야 함
y = boston.iloc[:,13]
# print(boston.head())


# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 6667)

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter = 'regressor')
# all_estimators => sklearn의 모든 classifier가 있음

# name, algorithm (변수명)으로 allAlgorithms이 반환값으로 반환함
# 
for (name, algroithm) in allAlgorithms:
    model = algroithm()


    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # print(name, "의 정답률 = ", accuracy_score(y_test, y_pred))
    score = model.score(x_test,y_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(name,"의 mse : ", mse)
    print(name,"의 r2 : ", r2)
    print(name,"의 score : ", score)

    print("======================")
#
    
# 26개의 모델을 한번에 돌림

print(sklearn.__version__)

'''
scikit-learn                       0.22.1 # 여기서는 안돌아가는 모델이 존재 (새로운 모델이 들어오고 안쓰는 모델이 빠지는 과정에서 안되는 모델 발생)

down grade ======>

scikit-learn                       0.20.1 # 모든 모델이 돌아가도록 다운그레이드

'''