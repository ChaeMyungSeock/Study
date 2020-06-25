from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.applications import VGG16
from keras import optimizers, initializers, regularizers, metrics

x_train, x_test, y_train, y_test = np.load('D:/Study/mini_proj/binary_image_data.npy',allow_pickle = True)
print(x_train.shape)
print(x_train.shape[0])
print(np.bincount(y_train))
print(np.bincount(y_test))

image_w = 36
image_h = 36
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255 

x_train = x_train.reshape(x_train.shape[0],36,36*3)
x_test = x_test.reshape(x_test.shape[0],36,36*3)

with K.tf_ops.device('/device:GPU:0'):
    # model = VGG16(weights = 'imagenet', include_top = True)

    model = Sequential()
    model.add(LSTM(250, input_shape=x_train.shape[1:], activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2048, kernel_initializer='he_normal'))
    model.add(Dense(1024, kernel_initializer='he_normal'))

    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/IU_Classify.model"
    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode = 'min')


history = model.fit(x_train, y_train,epochs=30, validation_split=0.2 , callbacks=[ early_stopping, checkpoint])

# , callbacks=[checkpoint, early_stopping]

loss, acc = model.evaluate(x_test, y_test)

print("acc : ", acc)
pred = model.predict(x_test)
print(pred)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()


    # model.add(Conv2D(64, (3,3), padding="same", input_shape=x_train.shape[1:], activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Conv2D(64, (3,3), padding="same",  activation="relu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    

    # model.add(Conv2D(128, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # # model.add(Conv2D(128, (3,3), padding="same",  activation="relu",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # # model.add(Conv2D(256, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # # model.add(Conv2D(256, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Conv2D(256, (3,3), padding="same",  activation="relu",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling2D(pool_size=(2,2)))


    # # model.add(Conv2D(512, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # # model.add(Conv2D(512, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Conv2D(512, (3,3), padding="same",  activation="relu",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # # model.add(Conv2D(512, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # # model.add(Conv2D(512, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Conv2D(512, (3,3), padding="same",  activation="relu",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Flatten())
    # # model.add(Dense(4096, kernel_initializer='he_normal'))
    # # model.add(Dense(4096, kernel_initializer='he_normal'))
    # # model.add(Dense(2048, kernel_initializer='he_normal'))
    # model.add(Dense(1024, kernel_initializer='he_normal'))
