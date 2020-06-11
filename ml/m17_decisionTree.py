from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=3 # random_state를 바꾸면 바뀜 왜냐하면 어떤행의 어떤 컬럼이 걸리냐에 따라서 달라지기 때문
)

model = RandomForestClassifier(max_depth=3) # default? 몇이 좋냐고?

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(acc)
print(model.feature_importances_)
# 0이 많음 => 영향가 없는 칼럼 (feature importance)