# 1.데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Deconvolution2D, Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

from keras.optimizers import Adam, RMSprop, SGD, Adadelta,Adagrad, Nadam, Adamax

# optimizer = Adam(learning_rate=0.001)

# optimizer = RMSprop(learning_rate=0.001)
optimizer = Adadelta(learning_rate=0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x, y, epochs=100)

loss = model.evaluate(x,y)

pred = model.predict([3.5])

print(loss)
print(pred)


'''
optimizer = RMSprop(learning_rate=0.001)

[0.10987918078899384, 0.10987918078899384]
[[3.2711961]]

optimizer = SGD(learning_rate=0.001)

[0.0027524596080183983, 0.0027524596080183983]
[[3.475449]]
'''