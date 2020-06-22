# 과적함 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization => dropout과 결과가 같음

from xgboost import XGBClassifier,XGBRegressor, plot_importance
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV

dataset = load_breast_cancer()

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
parameters = {"n_estimators":[1,10,100,1000], "learning_rate":[0.1, 0.01, 0.001], 
            "colsample_bytree":[0.6,0.7,0.8,0.9], "colsample_bylevel":[0.6,0.7,0.8,0.9]}
# 그리드 파라미터 사용.. 
# C : 1 kernel : lienear, C : 10 kernel : lienear, C : 100 kernel : lienear, C : 1000 kernel : lienear

kfold = KFold(n_splits=5, shuffle=True)

model = GridSearchCV(XGBClassifier(), parameters, cv=kfold)
# model = XGBRegressor()


model.fit(x_train, y_train)

score = model.score(x_test,y_test)

print('점수 : ', score)

# print(model.feature_importances_)
# 어떤 칼럼이 중요하진 한눈에 볼 수 있음

# print(b)

# plot_importance(model)
# plt.show()



'''

=========================
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.30000000000000004, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=800, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
=========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.6, 'learning_rate': 0.30000000000000004, 'n_estimators': 800}
=========================
PS D:\Study>  cd 'd:\Study'; & 'C:\Users\bitcamp\anaconda3\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2020.6.88468\pythonFiles\lib\python\debugpy\launcher' '55009' '--' 'd:\Study\ml\m23_xgb2_GridSearch.py'    
(569, 30)
(569,)
점수 :  0.9736842105263158
'''