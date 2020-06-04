from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 1. 데이터
x_data = [[0, 0], [1,0], [0,1], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# 모델은 한줄.. 파라미터값으로 늘어남

# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier(n_neighbors=1)
# model = KNeighborsClassifier() 최근접에 이웃을 파라미터로 넣어줘야 함 (n의 갯수는 성능차이) // 최근접 이웃 알고리즘
# 최근접 data의 선의 갯수 n_neighbors => n 데이터가 뭉쳐서 분류됨

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)


acc = accuracy_score([0,1,1,0], y_predict)

# accuracy_score => evaluate와 비슷

print(x_test, "의 예측 결과 : ", y_predict)
print("acc = ", acc)

#
