import os
import cv2
import numpy as np
from sklearn.utils import shuffle

x_save = np.load('D:/Study/mini_proj/images/train_data.npz')

# print(x_save.shape)
x = x_save['x']
y = x_save['y']

print(x)
print(y.shape)
