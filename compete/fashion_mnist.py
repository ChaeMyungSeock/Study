import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)


print(train_images.shape)
print(test_images.shape)

print(test_labels.shape)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.998):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()
model = keras.Sequential(
[tf.keras.layers.Conv2D(64,kernel_size =(3,3), input_shape=(28,28,1), activation='relu'),
tf.keras.layers.Conv2D(64,kernel_size =(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dropout(0.4),
tf.keras.layers.Dense(1024, activation='relu'),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels,validation_split=0.15, epochs=100,batch_size=12,callbacks=[callbacks])
prob_pred = model.predict(test_images)
prob_label = prob_pred.argmax(axis=-1)

np.savetxt('F:/Study/compet/data1/submit.txt', prob_label,fmt='%d')