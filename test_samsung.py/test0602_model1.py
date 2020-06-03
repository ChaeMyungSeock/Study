import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

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

print(samsung.shape)                    # (504, 6)
# print(samsung)
# print(hite.shape)                     # (504, 6, 5)
# print(hite)
x_hit = hite[5:510,:]
print(x_hit.shape)                      # (504,5)



x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)      #(504,5)
print(y_sam.shape)      #(504,)

# 2. 모델 구성

input1 = Input(shape=(5,))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)


input2 = Input(shape=(5,))
x2 = Dense(5)(input2)
x2= Dense(5)(x2)


merge = concatenate([x1, x2])

ouput = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = ouput)

model.summary()


# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss = 'mse')
model.fit([x_sam, x_hit], y_sam, epochs=5)

