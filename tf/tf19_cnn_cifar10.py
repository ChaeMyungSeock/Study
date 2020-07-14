from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import  np_utils
import pandas as pd
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train / 255
x_test = x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(x_train)
# print(y_train)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size) # 50000 / 100


x = tf.placeholder(tf.float32, [None, 3072])
x_img = tf.reshape(x, [-1, 32,32, 3])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) # dropout


w1 = tf.get_variable("w1", shape=[3,3,3,32] )
# [(a,b) ,c,d] => (a,b) => 커널사이즈 // c => 채널 // d => output node

L1 = tf.nn.conv2d(x_img, w1, strides = [1,1,1,1], padding='SAME')
# strides = [ 1,1,1,1] => [a,b,c,d] b,c의 경우만 일단 사용 a,d는 현재 상황에서는 형식상 사용한다고 생각
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],strides = [1,2,2,1], padding='SAME')



w2 = tf.get_variable("w2", shape=[3,3,32,64], )

L2 = tf.nn.conv2d(L1, w2, strides = [ 1,1,1,1], padding='SAME')
# strides = [ 1,1,1,1] => [a,b,c,d] b,c의 경우만 일단 사용 a,d는 현재 상황에서는 형식상 사용한다고 생각

L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1],strides = [1,2,2,1], padding='SAME')


print(L2)

L2_flat = tf.reshape(L2,[-1,8*8*64])

w3 = tf.get_variable("w", shape=[8*8*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
h3 = tf.nn.softmax(tf.matmul(L2_flat,w3)+b3)



cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(h3), axis = 1))




optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(cost)
# train = optimizer.minimize(cost)

prediction = tf.equal(tf.math.argmax(h3, 1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
with tf.compat.v1.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.compat.v1.global_variables_initializer())


    for epoch in range(training_epochs): # 15
        avg_cost = 0


        for i in range(total_batch):    # 600
            batch_xs, batch_ys = x_train[i*batch_size : (i*batch_size) + batch_size], y_train[i*batch_size: (i*batch_size)+ batch_size]
            feed_dict = {x_img:batch_xs, y:batch_ys, keep_prob : 1}
            c, _, acc_val= sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print("Epoch : ", "%4d" % (epoch+1) , "cost = " , "{:.9f}".format(avg_cost))

            
    feed_dict_test = {x_img:x_test, y:y_test, keep_prob : 0.7}

    acc = sess.run([accuracy], feed_dict = feed_dict_test)

        
    
    # h,p,a = sess.run([hypothesis,prediction, accuracy], feed_dict={x:x_test, y:y_test})
    print('훈련 끝')
    correct_prediction = tf.equal(tf.argmax(h3,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _, acc = sess.run([h3,accuracy], feed_dict={x_img: x_test, y: y_test, keep_prob : 0.6})
    
    print('Acc : ',acc ) ## acc 출력
# 

    
