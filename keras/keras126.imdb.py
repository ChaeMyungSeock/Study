'''
0과 1로 분류하는 데이터 셋
'''

from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)
# num_words 가져올 데이터의 종류 갯수

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value,key) for (key, value) in word_index.items()]
)
decode_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in x_train[0]]
)
# print(word_index)
# print(reverse_word_index)
print(decode_review)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))

# 카테고리 개수 출력
category = np.max(y_train)+1
print("카테고리 : ", category)

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train = pd.DataFrame(y_train)
bbb = y_train.groupby(0)[0].count()
print(bbb)
print(bbb.shape)



# 주간 과제 : groupby() 사용법 숙지

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=1000, padding='pre')
x_test = pad_sequences(x_test, maxlen=1000, padding='pre')
# maxlen 최대 넣을 데이터의 수 => 나머지값은 패딩으로 채워줌

# x_train = pad_sequences(x_train)

# truncating 자르겠다

# print(len(x_train[0]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape , x_test.shape)


# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
model.add(Embedding(2000,128))
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))

# model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]

print('acc', acc)



y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


'''
1. imdb검색해서 데이터 내용 확인.
2. word_size 전체데이터 부분 변경해서 최상값 확인
3. 주간과제: groupby()의 사용법 숙지
4. 인덱스를 단어로 바꿔주는 함수 찾을것
5. 125번 125번 튠
word_size = 2000    acc 0.8371800184249878


'''