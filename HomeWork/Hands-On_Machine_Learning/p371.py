import numpy as np
from sklearn.datasets import load_iris
from keras.layers import Flatten,Dense
from keras.models import Sequential
from sklearn.linear_model import Perceptron
import keras.datasets.fashion_mnist
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


print(x_train.shape)
print(x_train.dtype)

x_train = x_train / 255.0
x_test = x_test / 255.0


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]])
print(y_train[0])

# 2. 모델구성
model = Sequential()
model.add(Flatten(input_shape =(28,28)))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

hidden1 = model.layers[1]
weights, biass = hidden1.get_weights()
# print(weights)
# print(weights.shape)
# print(biass)
# print(biass.shape)

model.compile(loss = "sparse_categorical_crossentropy", metrics=["acc"], optimizer="SGD")
history = model.fit(x_train, y_train, epochs=30, validation_split = 0.2)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()