from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밋어요", "최고에요", "참 잘 만든 영화에요",
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밋네요']

# 긍정 1, 부정 0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])


# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index) # 중복 된 단어들은 앞 쪽으로 몰림 그리고 한번만 등장(인덱스 번호니까) => 많이 등장하는 놈이 맨 앞으로

x = token.texts_to_sequences(docs)
# print(x)

from keras.preprocessing.sequence import pad_sequences
# 패드 시퀀스 
'''
(2,) [3,7]
(1,) [2]
(3,) [4,5,11]
(5,) [5,4,3,2,6]

'''

pad_x_pre = pad_sequences(x, padding = 'pre')
print(pad_x_pre)        # (12,5)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 : ", word_size) # 25

print(pad_x_pre.shape)

# pad_x = pad_sequences(x, padding = 'post',value = 1.0)
# print(pad_x)

from keras.models import Sequential
from keras.layers import Embedding, Flatten,Dense,LSTM,Conv1D

model = Sequential()
model.add(Embedding(25,10, input_length=5))
# model.add(Embedding(25,10))

# word_size => 전체 단어수  10 => output 노드 
'''
Embedding()에 넣어야하는 대표적인 인자는 다음과 같다.
첫번째 인자 = 단어 집합의 크기. 즉, 총 단어의 개수
두번째 인자 = 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
input_length = 입력 시퀀스의 길이
'''
model.add(Conv1D(10,kernel_size=3, padding = 'valid',activation='relu' ,input_shape = (5,1)))
model.add(LSTM(3))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(optimizer='adam', metrics=['acc'], loss = 'binary_crossentropy')

model.fit(pad_x_pre, labels, epochs=30)

acc = model.evaluate(pad_x_pre, labels)[1]
print(acc)

pred = model.predict(pad_x_pre)
print(pred)