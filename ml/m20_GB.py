# 그라디언트 부스트 그럭저럭 쓸만함
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,GradientBoostingClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=3 # random_state를 바꾸면 바뀜 왜냐하면 어떤행의 어떤 컬럼이 걸리냐에 따라서 달라지기 때문
)

model = GradientBoostingRegressor(max_depth=3) # default? 몇이 좋냐고?

# max_features : 기본값 써라!
# n_estimators : 클수록 좋다!, 단점 메모리 짱 차지, 기본값 100
# n_jobs-1 : 병렬처리
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(acc)
print(model.feature_importances_)
# 0이 많음 => 영향가 없는 칼럼 (feature importance)

print(cancer.data.shape[1])
print(cancer.data.__class__)
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_caner(model):
    n_featrues = cancer.data.shape[1]
    plt.barh(np.arange(n_featrues), model.feature_importances_, align='center')
    # 가로 방향으로 바차트를 그림
    plt.yticks(np.arange(n_featrues), cancer.feature_names)
    # 축의 틱과 축의 틱 라벵르 편집할 수 있음
    # yticks(ticks=None, labels=None,**kwargs)
    # ticks 틱이 위치하는 리스트, 각 축의 틱을 사용하지 않기위해 빈 리스트 입력도 가능
    # 설정한 위치에 표시할 라벨을 선정한다
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.ylim(-1,n_featrues)

plot_feature_importances_caner(model)
plt.show()
    