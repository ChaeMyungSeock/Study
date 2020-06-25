from keras import backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = np.load('D:/Study/mini_proj/binary_image_data.npy',allow_pickle = True)


image_generator = ImageDataGenerator(
    rotation_range = 300,
    width_shift_range = 0.5,
    height_shift_range = 0.5,
    zoom_range = 0.5,
    horizontal_flip=True,
    fill_mode = 'nearest',
)

xtas, ytas = [], []
for i in range(x_test.shape[0]):
    num_aug = 0
    x = x_test[i]
    x = x.reshape((1,) + x.shape)
    for x_aug in image_generator.flow(x, batch_size = 1, save_to_dir='D:/Study/mini_proj/new_test', save_prefix='test', save_format='png') :
        if num_aug >= 40:
            break
        xtas.append(x_aug[0])
        num_aug += 1