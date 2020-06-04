from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 로지스틱 디그래서 => 분류 좀 있다 확인

# 1. 데이터
dataset = load_iris()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델
# 모델은 한줄.. 파라미터값으로 늘어남

model = RandomForestRegressor()  # score : 0.96625, 알아서 파라미터 튜닝이 알아서 들어감
# error : Classification metrics can't handle a mix of multiclass and continuous targets => accuracy_score 오류 (분류인데 회기 쓸 때)
# model = LinearSVC() # acc = 0.8666
# model = SVC() # acc = 0.9
# model = KNeighborsClassifier(n_neighbors=2) # acc = 0.933 
# model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456) # acc = 0.9
# model = KNeighborsClassifier() 최근접에 이웃을 파라미터로 넣어줘야 함 (n의 갯수는 성능차이) // 최근접 이웃 알고리즘
# 최근접 data의 선의 갯수 n_neighbors => n 데이터가 뭉쳐서 분류됨
# model = KNeighborsRegressor()   # score : 0.976
# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측

# y_predict = model.predict(x_test)

score = model.score(x_test,y_test)

# acc = accuracy_score(y_test, y_predict)

# accuracy_score => evaluate와 비슷

# print(x_test, "의 예측 결과 : \n", y_predict)
# print("acc = ", acc)
#
#
# r2 = r2_score(y_test, y_predict)

print("score : ", score)
# print("r2 : ", r2)
