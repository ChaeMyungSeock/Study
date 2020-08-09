import numpy as np
import pandas as pd

train_data = pd.read_csv('./data/dacon/comp_vision/train.csv', header=0, index_col=None)
test_data = pd.read_csv('./data/dacon/comp_vision/test.csv', header=0, index_col=None)
# print(train_data.head())
# print(test_data.head())

data_digit = train_data['digit']
data_letter = train_data['letter']
data_train_id = train_data['id']

# data_digit = test_data['digit']
data_test_letter = test_data['letter']
data_test_id = test_data['id']

# print(data_digit)
# print(data_letter)
# print(data_test_letter)
# print(train_data.head())

# print(train_data.head)

del train_data['digit']
del train_data['letter']
del train_data['id']

del test_data['letter']
del test_data['id']
# print(train_data.head())
# print(test_data.head())

train = train_data.values
test = test_data.values
train_ex = data_digit.values
train_ex1 = data_letter.values

print(train_ex)
# print(train_ex1)

# print(train.shape)
# print(test.shape)

# np.save('./data/dacon/comp_vision/data_npy/train_data.npy',train)
# np.save('./data/dacon/comp_vision/data_npy/test_data.npy',test)
np.save('./data/dacon/comp_vision/data_npy/train_target_data.npy',train_ex)
# np.save('./data/dacon/comp_vision/data_npy/test_data.npy',test_target)
