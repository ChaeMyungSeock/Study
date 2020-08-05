from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
import random
def autoencoder(hidden_layer_size):

    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_train = x_train / 255.
x_test = x_test / 255.
model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_03 = autoencoder(hidden_layer_size=4)
model_04 = autoencoder(hidden_layer_size=8)
model_05 = autoencoder(hidden_layer_size=16)
model_06 = autoencoder(hidden_layer_size=32)

model_01.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
model_01.fit(x_train, x_train, epochs=10)

model_02.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
model_02.fit(x_train, x_train, epochs=10)


model_03.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
model_03.fit(x_train, x_train, epochs=10)


model_04.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
model_04.fit(x_train, x_train, epochs=10)


model_05.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
model_05.fit(x_train, x_train, epochs=10)


model_06.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
model_06.fit(x_train, x_train, epochs=10)


# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc']) # loss 0.0102
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093
# model.fit(x_train, x_train, epochs=10)

# output = model.predict(x_test)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_03 = model_03.predict(x_test)
output_04 = model_04.predict(x_test)
output_05 = model_05.predict(x_test)
output_06 = model_06.predict(x_test)


# 그림을 그리자
fig, axes = plt.subplots(7, 5, figsize = (15,15))

random_imgs = random.sample(range(output_01.shape[0]),5)
outputs = [ x_test, output_01, output_02, output_03, output_04,
            output_05, output_06]


for row_num, row in enumerate(axes):
    for col_num, col in enumerate(row):
        col.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap = 'gray')

        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
plt.show()