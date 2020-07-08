import tensorflow as tf
tf.set_random_seed(777)

x_data =    [[73, 51, 65],
            [92, 98, 11],
            [89,31,33],
            [99, 33, 100],
            [17,66,79]]

y_data = [[152],
         [185],
         [180], 
         [205],
        [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])


w = tf.Variable(tf.random_normal([3,1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias',dtype = tf.float32)

hypothesis = tf.matmul(x,w) + b

# tf.matmul(x,w) => w*x  x => (5.3)  w => (3,1) w*x => (5,1) 행렬연산을 도와줌

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 7.5e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict = {x : x_data, y: y_data})


    if step % 20 == 0:
        print(step, 'cost : ', cost_val , '\n 예측값 : ', hy_val)

sess.close()