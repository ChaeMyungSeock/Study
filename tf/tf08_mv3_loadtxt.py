import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(777)

dataset = np.loadtxt('./data/csv/data-01-test-score.csv',delimiter=',')

x_data = dataset[:,0:-1]
y_data = dataset[:,[-1]]


'''
y_data = dataset[:,[-1]]

[[152.]
 [185.]
 .
 .
 .
 [142.]
 [101.]
 [149.]]


y_data = dataset[:,-1]
[152. 185. 180. 196. 142. 101. 149. 115. 175. 164. 141. 141. 184. 152.
 148. 192. 147. 183. 177. 159. 177. 175. 175. 149. 192.]

'''

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])


w = tf.Variable(tf.random_normal([3,1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), name = 'bias',dtype = tf.float32)

hypothesis = tf.matmul(x,w) + b

# tf.matmul(x,w) => w*x  x => (5.3)  w => (3,1) w*x => (5,1) 행렬연산을 도와줌

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict = {x : x_data, y: y_data})


    if step % 20 == 0:
        print(step, 'cost : ', cost_val , '\n 예측값 : ', hy_val)

sess.close()
