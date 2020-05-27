# 2번의 첫번째 답

import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y-1

print(y)

from keras.utils import np_utils
# y = np


# 2번의 두번째 답
y = np.array([1,2,3,4,5,1,2,3,4,5])
 # (10,)
# y = y.reshape(-1, 1)
y = y.reshape(10,1)
print(y)
print(y.shape) 
from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)


y = aaa.transform(y).toarray()

print(y)
print(y.shape)
