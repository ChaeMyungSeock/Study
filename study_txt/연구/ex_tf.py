import tensorflow as tf

print(tf.__version__)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())