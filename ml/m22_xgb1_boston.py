# 과적함 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization => dropout과 결과가 같음

from xgboost import XGBClassifier,XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = load_boston()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

n_estimators = 90
learning_rate = 0.1229       # 딥러닝에서 loss, 옵티마이저 부분임 // 학습률 
colsample_bytree = 0.81      # 우승모델의 경우 0.6~0.9 사용 default 값은 1 // sample을 얼마나 활용할 것인가
colsample_bylevel = 0.6     # 
# cv써서 결과값 봐야 함

n_jobs = -1 # 병렬처리 머신에서는 활용하자
max_depth = 5

'''
트리구조 특징 => 전처리안해도 됨, 결측치 제거를 안해도 된다
xgb => 딥러닝에 비해 빠름 but 다른 머신러닝에 비해선 조금 느림
'''

model = XGBRegressor(max_depth = max_depth, learning_rate = learning_rate, n_estimators = n_estimators,
                     n_jobs = n_jobs, colsample_bylevel = colsample_bylevel, colsample_bytree = colsample_bytree)
# model = XGBRegressor()


model.fit(x_train, y_train)

score = model.score(x_test,y_test)

print('점수 : ', score)

print(model.feature_importances_)
# 어떤 칼럼이 중요하진 한눈에 볼 수 있음


plot_importance(model)
# plt.show()





# n_estimators = 1000
# learning_rate = 0.01        # 딥러닝에서 loss, 옵티마이저 부분임 // 학습률 
# colsample_bytree = 0.9      # 우승모델의 경우 0.6~0.9 사용 default 값은 1 // sample을 얼마나 활용할 것인가
# colsample_bylevel = 0.9 
# 점수 :  0.9356198812681292



# n_estimators = 90
# learning_rate = 0.1229       # 딥러닝에서 loss, 옵티마이저 부분임 // 학습률 
# colsample_bytree = 0.81      # 우승모델의 경우 0.6~0.9 사용 default 값은 1 // sample을 얼마나 활용할 것인가
# colsample_bylevel = 0.6  
# 점수 :  0.9508500083430775