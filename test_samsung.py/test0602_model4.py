import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# concatenate 함수
# Concatenate 클래스
def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    return np.array(aaa)

size = 6 # 앞에 5개 x 뒤에 하나 y

# 1. 데이터
# npy 불러오기
samsung = np.load('./data/samsung0603.npy', allow_pickle='True')
hite = np.load('./data/hite0603.npy',allow_pickle='True')

print(samsung.shape)                     # (509, 1)
print(hite.shape)                        # (509, 5)

samsung = samsung.reshape(samsung.shape[0],)

samsung = (split_x(samsung, size))      # split일 때 vector형태로 넣었는데 지금은 (1, 2, 3) 이 아니라 [1], [2], [3] => 형태로 나눠지니깐 reshape를 해줘야함...
# hite = (split_x(hite, size))


# print(samsung.shape)                    # (504, 6)
# print(hite.shape)                    # (504, 6, 5)

scaler = StandardScaler()
scaler.fit(hite)
hite = scaler.transform(hite)
pca = PCA(n_components= 1)
pca.fit(hite)
hite = pca.transform(hite)
scaler.fit(samsung)
samsung = scaler.transform(samsung)
# print(hite.shape)
hite = (split_x(hite, size)) 
# print(hite.shape)


# print(samsung)
# print(hite.shape)                     # (504, 6, 5)
# print(hite)
# x_hit = hite[5:510,:]
# print(x_hit.shape)                      # (504,5)


x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

x_sam = x_sam.reshape(x_sam.shape[0],5,1)
hite = hite.reshape(hite.shape[0],hite.shape[1]*hite.shape[2])

x_sam_train, x_sam_test, y_sam_train, y_sam_test = train_test_split(x_sam,y_sam, test_size =0.2,)
hite_train, hite_test = train_test_split(hite, test_size = 0.2)

# print(x_sam.shape)      #(504,5,1)
# # y_sam = y_sam.reshape(y_sam.shape[0],1)
# print(y_sam.shape)      #(504,)

print(x_sam_train.shape)
print(hite_train.shape)
print(x_sam_test.shape)
print(hite_test.shape)

# x_sam_train ,x_sam_test, y_sam_trian, y_sam_test


# 2. 모델 구성

input1 = Input(shape=(5,1))
x1 = LSTM(100)(input1)
x1 = Dense(100)(x1)


input2 = Input(shape=(6,))
x2 = Dense(100)(input2)
x2= Dense(150)(x2)


# merge = concatenate([x1, x2])
merge = Concatenate()([x1,x2])

dense1 = Dense(50)(merge)
ouput = Dense(1)(dense1)
model = Model(inputs = [input1, input2], outputs = ouput)

model.summary()


# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss = 'mse')
model.fit([x_sam_train, hite_train], y_sam_train, epochs=10,validation_split=0.2,batch_size=1)

# 4. 평가
# loss, mse = model.evaluate(x0_test,y0_test,batch_size=1)

print(x_sam.shape)
y_predict = model.predict([x_sam_test,hite_test])

# print('y_pre : ', y_predict)

# print(y_predit.shape)

y_predict1 = scaler.inverse_transform(y_predict)
print("y_pre : ", y_predict1)

    
# RMSE 구하기
from sklearn.metrics import mean_squared_error #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_sam_test, y_predict))

# R2구하기 0~1 사이의 값 1에 가까울수록 신뢰도가 올라감 but 맹신은 금지 (데이터의 연관성이 있긴 하나 다른 데이터의 변수도 생각해야함)
from sklearn.metrics import r2_score
r2 = r2_score(y_sam_test, y_predict)
print("R2 : ", r2)
