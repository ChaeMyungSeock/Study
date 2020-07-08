from sklearn.datasets import load_breast_cancer
import tensorflow as tf

dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target

print(x_data.shape)
print(y_data.shape)

y_data = y_data.reshape(-1,1)
print(y_data.shape)
feature = x_data[1].shape
# print(feature)
x = tf.compat.v1.placeholder(dtype = tf.float32,shape=  [None, 30])
y = tf.compat.v1.placeholder(dtype= tf.float32, shape= [None, 1])

w = tf.Variable(tf.zeros([30,1]), name = 'weight', dtype = tf.float32)
b = tf.Variable(tf.zeros([1]), name = 'bias', dtype = tf.float32)

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y)* tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1.2e-7)
train = optimizer.minimize(cost)
predicted = tf.cast(hypothesis >0.5, dtype=tf.float32)
accurcy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

# mse = mean_squared_error()

with tf.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.global_variables_initializer())

    for step in range(8001):
            cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})

            if step % 20 ==0:
                print(step, cost_val)
    h,c,a = sess.run([hypothesis, predicted, accurcy], feed_dict={x:x_data, y:y_data})

    print('\n Hypothesis : ', h , '\n Accuracy : ', a)