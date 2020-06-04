from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 로지스틱 디그래서 => 분류 좀 있다 확인

# 1. 데이터
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 66)


# 2. 모델
# 모델은 한줄.. 파라미터값으로 늘어남

# model = RandomForestRegressor()  # score :  0.843976545272302
# error : Classification metrics can't handle a mix of multiclass and continuous targets => accuracy_score 오류 (분류인데 회기 쓸 때)
# model = LinearSVC() # score :  0.843976545272302 acc =  0.8947368421052632
# model = SVC() # acc =  0.8947368421052632 score :  0.8947368421052632
# model = KNeighborsClassifier(n_neighbors=2) # acc =  0.8947368421052632 score :  0.8947368421052632
# model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)  #acc =  0.956140350877193  score :  0.956140350877193
# model = KNeighborsClassifier() # acc =  0.9210526315789473 score :  0.9210526315789473
# model = KNeighborsRegressor()      # score :  0.7074774473772133

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

# acc = accuracy_score(y_test, y_predict)
#
# accuracy_score => evaluate와 비슷

print(x_test, "의 예측 결과 : ", y_predict)
# print("acc = ", acc)

#
print("score : ", score)