from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 로지스틱 디그래서 => 분류 좀 있다 확인
# 스케일링 할 때 각 칼럼별로 스케일링 되기 때문에 각 가격 판매량 칼럼별로 수행 큰 차이가 없다.

# 1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 777)
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler이후에 pca로 차원 축소에서 시간 확보 기준점 정확도 확보
# feature important로 뽑은걸로 비교
# 랜덤포레스트  = > 100개중에 5개 중요한거 뽑아줌
# 중요한 feature를 다 다르게 보여줌 



# 2. 모델
# 모델은 한줄.. 파라미터값으로 늘어남

# model = RandomForestRegressor() #mse :  8.933848382352943 r2 :  0.8834666690212107 score :  0.8834666690212108

# model = LinearSVC()  # error : Unknown label type: 'continuous

model = SVC()   # error : Unknown label type: 'continuous

# model = KNeighborsClassifier(n_neighbors=2) # error : Unknown label type: 'continuous'

# model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456) error :Unknown label type: 'continuous'

# model = KNeighborsRegressor()   # mse :  21.651090196078435 r2 :  0.7175826640560564 score :  0.7175826640560564

# model = KNeighborsClassifier()  # error : Unknown label type: 'continuous'
# 최근접에 이웃을 파라미터로 넣어줘야 함 (n의 갯수는 성능차이) // 최근접 이웃 알고리즘 최근접 data의 선의 갯수 n_neighbors => n 데이터가 뭉쳐서 분류됨

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측

y_predict = model.predict(x_test)
score = model.score(x_test,y_test)


# acc = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
# accuracy_score => evaluate와 비슷

print("bost 집값의 예측 결과 : ", y_predict)
print("mse : ", mse)
print("r2 : ", r2)
#
print("score : ", score)
