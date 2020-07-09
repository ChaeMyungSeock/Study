import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train)


sess = tf.compat.v1.Session()
y_train = tf.one_hot(y_train,depth=10).eval(session=sess)
y_test = tf.one_hot(y_test,depth=10).eval(session=sess)
sess.close()


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]* x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2])

print(x_train.shape)
print(y_train.shape)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])


# w = tf.Variable(tf.random_normal([784,512]), name = 'weight', dtype = tf.float32)
# b = tf.Variable(tf.zeros([512]), name = 'bias',dtype = tf.float32)
# layer = tf.nn.softmax(tf.matmul(x,w) + b)



w1 = tf.Variable(tf.random.normal([784, 1024]), name = "weight1", dtype = tf.float32)
b1 = tf.Variable(tf.zeros([1024]), name = 'bias1',dtype = tf.float32)
layer1 = tf.matmul(x,w1) + b1
# model.add(Dense(50))

w2 = tf.Variable(tf.random.normal([1024, 512]), name = "weight2", dtype = tf.float32)
b2 = tf.Variable(tf.zeros([512]), name = 'bias2',dtype = tf.float32)
layer2 = tf.matmul(layer1,w2) + b2
# model.add(Dense(50))

w3 = tf.Variable(tf.random.normal([512, 256]), name = "weight3", dtype = tf.float32)
b3 = tf.Variable(tf.zeros([256]), name = 'bias3',dtype = tf.float32)
layer3 = tf.matmul(layer2,w3) + b3
# model.add(Dense(50))


w4 = tf.Variable(tf.random.normal([256, 128]), name = "weight4", dtype = tf.float32)
b4 = tf.Variable(tf.zeros([128]), name = 'bias4',dtype = tf.float32)
layer4 = tf.matmul(layer3,w4) + b4
# model.add(Dense(50))

w5 = tf.Variable(tf.random.normal([128, 64]), name = "weight5", dtype = tf.float32)
b5 = tf.Variable(tf.zeros([64]), name = 'bias5',dtype = tf.float32)
layer5 = tf.matmul(layer4,w5) + b5
# model.add(Dense(50))

w6 = tf.Variable(tf.random.normal([64, 32]), name = "weight6", dtype = tf.float32)
b6 = tf.Variable(tf.zeros([32]), name = 'bias6',dtype = tf.float32)
layer6 = tf.nn.softmax(tf.matmul(layer5,w6) + b6)
# model.add(Dense(50))

w7 = tf.Variable(tf.random.normal([32, 10]), name = "weight7", dtype = tf.float32)
b7 = tf.Variable(tf.zeros([10]), name = 'bias7',dtype = tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(layer6,w7) + b7)
# model.add(Dense(50)


loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis = 1))


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 1e-7)
train = optimizer.minimize(loss)

predicted = tf.math.argmax(hypothesis,1)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))


with tf.compat.v1.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        loss_val, _, acc_val = sess.run([loss, train, acc], feed_dict={x: x_train, y: y_train})

        if step % 20 ==0:
            print(step, loss_val, acc_val)
    h,c,a = sess.run([hypothesis, predicted, acc], feed_dict={x:x_test})
    

    print('\n Hypothesis : ', h ,'\n Correct (y) : ', c , '\n Accuracy : ', a)