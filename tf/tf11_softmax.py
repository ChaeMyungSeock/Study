import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


x = tf.placeholder(shape = (None, 4), dtype = tf.float32, name = 'x_data')
y = tf.placeholder(shape = (None, 3), dtype = tf.float32, name = 'y_data')

w = tf.Variable(tf.random_normal([4,3]), name = 'weight1', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1,3]), name = 'bias1', dtype = tf.float32)

hypothesis = tf.nn.softmax(tf.matmul(x,w)+b)

# matmul => matrix multi => 매트릭스 곱

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(10001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: x_data, y:y_data} )
        
        if step % 200 == 0:
            print(step, loss_val)
    
    a = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.math.argmax(a,1)))
    
    b = sess.run(hypothesis, feed_dict={x: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.math.argmax(b,1)))

    c = sess.run(hypothesis, feed_dict={x: [[3, 5, 7, 11]]})
    print(c, sess.run(tf.math.argmax(c,1)))

    d = sess.run(hypothesis, feed_dict={x: [[11, 33, 4, 13]]})
    print(d, sess.run(tf.math.argmax(d,1)))


    # all = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9],
    #                                         [1, 3, 4, 3],
    #                                         [3, 5, 7, 11],
    #                                         [11, 33, 4, 13]]})
    feed_dict={x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]}
    all = sess.run(hypothesis, feed_dict = feed_dict)

    print("all의 예측값", all, "all의 예측값 중 최대값: ", sess.run(tf.argmax(all,1)))# 최적의 w와 b가 구해져 있다