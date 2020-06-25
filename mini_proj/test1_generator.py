from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

x_train, x_test, y_train, y_test = np.load('D:/Study/mini_proj/binary_image_data.npy',allow_pickle = True)
print(x_train.shape)
print(x_test.shape)

print(x_train.shape[0])
print(np.bincount(y_train))
print(np.bincount(y_test))
print(y_train.shape)

image_w = 36
image_h = 36
# print(len(x_train))
for i in range(len(x_test)):
    x1_test = x_test[i,:,:,:]
    img = Image.fromarray(x_test, 'RGB')
    img.save(str(i)+'.jpg')
'''
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


with K.tf_ops.device('/device:GPU:0'):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding="same", input_shape=x_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/dog_cat_classify.model"
    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', patience=10, mode = 'min')


history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2 , callbacks=[checkpoint, early_stopping]
)

# , callbacks=[checkpoint, early_stopping]

print("정확도 : %.2f " %(model.evaluate(x_test, y_test)[1]))



# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')
# plt.show()

'''