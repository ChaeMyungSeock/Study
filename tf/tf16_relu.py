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


'''
layer1 = tf.nn.relu(matmul(x,w1) + b1)
layer1 = tf.nn.elu(matmul(x,w1) + b1)
layer1 = tf.nn.selu(matmul(x,w1) + b1)

'''





w1 = tf.Variable(tf.zeros([784, 512]), name = "weight1", dtype = tf.float32)
b1 = tf.Variable(tf.zeros([512]), name = 'bias1',dtype = tf.float32)
layer1 =  tf.nn.relu(tf.matmul(x,w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob= 0.2)
# model.add(Dense(50))
'''
w3 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1), name='weight1')
b3 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias1')
layer3 = tf.nn.selu(tf.matmul(layer2,w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.2)
'''

w3 = tf.Variable(tf.zeros([512, 10]), name = "weight3", dtype = tf.float32)
b3 = tf.Variable(tf.zeros([10]), name = 'bias3',dtype = tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(layer1,w3) + b3)
# model.add(Dense(50)


loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis = 1))


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.4)
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