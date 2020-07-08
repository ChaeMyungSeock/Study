from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


dataset = load_diabetes()

data_x = dataset.data
data_y = dataset.target
features = data_x.shape[1]
data_y = data_y.reshape(-1,1)
print(data_x.shape)
print(data_y.shape)
print(features)


# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 666 )


x = tf.compat.v1.placeholder(dtype = tf.float32,shape=  [None, features])
y = tf.compat.v1.placeholder(dtype= tf.float32, shape= [None, 1])


w = tf.Variable(tf.random_normal([features,1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias',dtype = tf.float32)

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.losses.mean_squared_error(data_y,hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(cost)

# mse = mean_squared_error()

with tf.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _= sess.run([cost, train], feed_dict={x: data_x, y: data_y})

        if step % 200 ==0:
            print(step, cost_val)