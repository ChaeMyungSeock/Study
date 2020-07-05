from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.layers import Flatten,Dense
from keras.models import Sequential
from sklearn.linear_model import Perceptron
import keras.datasets.fashion_mnist
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()

x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(30, activation = 'relu', input_shape = x_train.shape[1:]))
model.add(Dense(1))

model.compile(loss = "mse", optimizer = 'sgd')
history = model.fit(x_train, y_train, epochs = 20, validation_data = (x_val, y_val))

mse_test = model.evaluate(x_test,y_test)
x_new = x_test[:3]
y_pred = model.predict(x_new)

print(y_pred)