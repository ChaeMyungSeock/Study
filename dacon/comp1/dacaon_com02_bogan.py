import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D,Conv1D,MaxPool1D,Dropout
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)
# LSTM형식인데 시계열인지 명확하지 않으면 CNN으로 해야함 시계열 형식이 375단위로 잘리기 때문
# why? 잘라서 특징을 추출하기 때문

train = pd.read_csv('./data/dacon/comp2/train_features.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp2/train_target.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header=0, index_col=0)
x_prdeict = pd.read_csv('./data/dacon/comp2/test_features.csv', header=0, index_col=0)

# print(train.head())
# print(test.head())
# print(x_prdeict.head())
# print(submission.head())

print('train.shape :', train.shape) # (10500000,75) : x_train, test
print('test.shape :', test.shape)   # (2800,4) : x_predict
print('x_predict.shape :', x_prdeict.shape) # (262500,75) : x_train, test
print('submmission.shape :', submission.shape)  # (700,4)     : y_predict
train = train.values
x_predict = x_predict.values

train = train.reshape(2800,375,5)
x_prdeict = x_prdeict.reshape(700,375,5)


# def point(dataset):
#     for i in range(2800):
    
#     aaa = np.zeros((1,375,4),dtype=np.float64)
#     aaa = train[i,:,:]
#     aaa = aaa.reshape(375,4)


print(train)
