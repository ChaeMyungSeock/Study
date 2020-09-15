import time
import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
import matplotlib.pyplot as plt

start = time.time()
print(f'Start time : {start}')
# data
x_train = np.load('D:/study/data/train_val.npy')
y_train = np.loadtxt('D:/study/data/train_val_label.txt')

x_test = np.load('D:/study/data/test.npy')
y_test = np.loadtxt('D:/study/data/test_label.txt')
x_train = x_train[:10000]
y_train = y_train[:10000]
print(f'Train of feature : {x_train.shape}')
print(f'Test of feature : {y_train.shape}')
print(f'Train of Label : {x_test.shape}')
print(f'Test of Label : {y_test.shape}')

x_test = x_test[:5138]
y_test = y_test[:5138]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# model
BATCH_SIZE = 64
EPOCHS = 10

model = tf.compat.v1.keras.models.Sequential([
    EfficientNetB0(include_top=False, input_shape=(224,224,3), pooling='avg'),
    tf.compat.v1.keras.layers.Dense(20),
    tf.compat.v1.keras.layers.BatchNormalization(),
    tf.compat.v1.keras.layers.Activation('softmax')
])

# fitting
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(lr=0.01),
              metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs=EPOCHS,
                 batch_size=BATCH_SIZE, validation_split=0.2)

# evaluate
res = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f'loss : {res[0]}')
print(f'acc  : {res[1]}')
print(f'End time : {time.time() - start}')

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='^', c='magenta', label='loss')
plt.plot(hist.history['val_loss'], marker='^', c='cyan', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['accuracy'], marker='^', c='magenta', label='acc')
plt.plot(hist.history['val_accuracy'], marker='^', c='cyan', label='val_acc')
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.show()