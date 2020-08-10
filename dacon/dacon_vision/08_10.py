import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import layers
from keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,BatchNormalization,Lambda, AveragePooling2D,Dropout
from keras import models
train = pd.read_csv("./data/dacon/comp_vision/train.csv", index_col=0)
test = pd.read_csv("./data/dacon/comp_vision/test.csv", index_col=0)
submission = pd.read_csv("./data/dacon/comp_vision/submission.csv", index_col=0)

x_train = np.load('./data/dacon/comp_vision/data_npy/train_data.npy',allow_pickle=True)
x_test = np.load('./data/dacon/comp_vision/data_npy/test_data.npy',allow_pickle=True)
y_train = np.load('./data/dacon/comp_vision/data_npy/train_target_data.npy',allow_pickle=True)

# print(x_train.shape)
# print(x_test.shape)
# x_train = x_train.reshape(x_train.shape[0],28,28)
# plt.imshow(x_train[3])
# plt.gray()

# plt.show()

x_train = x_train.reshape(x_train.shape[0],28,28,1)
y_train = to_categorical(y_train)
# print(x_train.shape)
# print(y_train.shape)
# print(x_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1,random_state = 66, shuffle = True)

train_datagener = ImageDataGenerator(rescale = 1./255,
    rotation_range=60,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip = True,
    fill_mode='nearest',
    )
test_datagener = ImageDataGenerator(rescale = 1./255)


model = models.Sequential()
model.add(layers.Conv2D(64, kernel_size=3, activation='relu',
                        input_shape=(28, 28, 1)))
model.add(layers.Dropout(0.25))

# model.add(BatchNormalization())

model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(layers.Dropout(0.25))
# model.add(BatchNormalization())

model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
model.add(layers.Dropout(0.25))
# model.add(BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
model.add(layers.Dropout(0.25))
# model.add(BatchNormalization())
model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(512, kernel_size=5, activation='relu'))
# model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dropout(0.6))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()



# model = models.Sequential()
# model.add(Conv2D(32, kernel_size=3, activation='relu',
#                         input_shape=(28, 28, 1)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Conv2D(256, kernel_size=3, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()


model.compile(optimizer=Adam(learning_rate=2e-4), loss="categorical_crossentropy", metrics=["accuracy"])
batch = 25
epoch = 250
history = model.fit(
    train_datagener.flow(x_train, y_train, batch_size=batch),
    steps_per_epoch=x_train.shape[0]//batch, # 전체 데이터 수 / 배치 사이즈
    epochs=epoch,
    validation_data=test_datagener.flow(x_val, y_val),
    verbose=1,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'bo', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# loss, acc1 = model.evaluate(x_train,x_test)
x_test = np.array(test.iloc[:, 1:]).reshape(-1, 28, 28, 1).astype(np.float)
x_test /= 255.
print(x_test.shape)
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
# print(acc1)
print(pred.shape)
print(pred[:5])

submission.digit = pred
submission.head()

submission.to_csv('./dacon/dacon_vision/submission/0811_submission_3.csv') # Accuracy: 0.53921

