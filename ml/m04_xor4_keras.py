from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import LeakyReLU
import numpy as np
# 1. 데이터
x_data = [[0, 0], [1,0], [0,1], [1,1]]

y_data = [0, 1, 1, 0]
x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape)
print(y_data.shape)

# y_data = y_data.reshape(y_data.shape[0],1)

print(x_data.shape)
print(y_data.shape)

# 2. 모델
# 모델은 한줄.. 파라미터값으로 늘어남

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)
# model = KNeighborsClassifier() 최근접에 이웃을 파라미터로 넣어줘야 함 (n의 갯수는 성능차이) // 최근접 이웃 알고리즘
model = Sequential()
model.add(Dense(10,input_shape=(2,),activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. 훈련
model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=1000)

# 4. 평가 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
x_test = np.array(x_test)
y_predict = model.predict(x_test)

loss, acc = model.evaluate(x_test, y_data)
# acc = accuracy_score([0,1,1,0], y_predict)

# accuracy_score => evaluate와 비슷
# y_predict = np.around(y_predict)

print(x_test, "의 예측 결과 : ", y_predict)
print("acc ", acc)

