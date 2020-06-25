import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import glob
import cv2.cv2 as cv2
import time
import os
from PIL import Image
from keras.models import Sequential 
from keras.layers import Conv2D , Dropout
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense
# from sklearn.ensemble import 

np.random.seed(15)

data_generator = ImageDataGenerator(rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''
이미지 데이터 픽셀 값은 0~255범위 값을 가짐 0~1사이로 정규화
서브 디렉토리의 폴더명이 해당 폴더에 들어있는 이미지들의 라벨이 됨

'''
batch_size = 4
iteration = 5
images=[]
# print(images.__class__)
# print(images)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), padding='same', input_shape = (100, 100, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2),padding='same'))

# # Adding a second convolutional layer
# classifier.add(Conv2D(32, (3, 3), activation = 'relu',padding='same'))
# classifier.add(MaxPooling2D(pool_size = (2, 2),padding='same'))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])


# print(cv_img)

train_generator = data_generator.flow_from_directory(
    'C:/Users/bitcamp/Desktop/IU/',
    target_size=(100,100),
    batch_size=3,
    class_mode='binary')
print(len(train_generator))

test_generator = data_generator.flow_from_directory(
    'D:/Study/mini_proj_testdata/',
    target_size=(100,100),
    batch_size=3,
    class_mode='binary')
print(len(test_generator))
images.append(test_generator)
print(images)

classifier.fit_generator(train_generator,
                         steps_per_epoch = 300,
                         epochs = 2,
                         validation_data = test_generator,
                         validation_steps = 5)

predict = classifier.predict_generator(test_generator, steps=5)
loss, acc = classifier.evaluate_generator(train_generator, steps=5)
# print(train_generator.class_indices)
print("loss : ", loss)
print("acc : ",acc)
print(predict)

# train_generator = np.array(train_generator)
# print(train_generator)
# test_generator = np.array(test_generator)
# print(test_generator.shape)

# x_train, y_train = train_generator.next()
# print(x_train.shape)
# print(y_train.shape)
# x_train = x_train.reshape(100,100,3)
# plt.imshow(x_train[0])
# plt.show()

