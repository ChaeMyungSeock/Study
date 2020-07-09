# 다중분류
# iris 코드를 완성하시오

from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.model_selection import train_test_split
dataset = load_iris()
x_data = dataset.data
y_data = dataset.target

print(y_data.shape)
# print(y_data)

sess = tf.compat.v1.Session()

y_data = tf.one_hot(y_data,depth=3).eval(session=sess)
sess.close()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

print(x_train.shape)
print(y_train.shape)

x1 = tf.compat.v1.placeholder(shape=(None, 4), dtype=tf.float32, name = 'train_x')
y1 = tf.compat.v1.placeholder(shape=(None, 3), dtype=tf.float32, name = 'train_y')



w = tf.Variable(tf.random_normal([4,3]), name = 'weight1', dtype = tf.float32)
b = tf.Variable(tf.zeros([1,3]), name = 'bias1', dtype = tf.float32)


hypothesis = tf.nn.softmax(tf.matmul(x1,w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y1*tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)

# predicted = tf.cast(tf.math.argmax(hypothesis,1), dtype=tf.float32)
# test = tf.cast(tf.math.argmax(y_test,1), dtype=tf.float32)

# accurcy = tf.reduce_mean(tf.cast(tf.equal(test,predicted), dtype=tf.float32))

# prediction = tf.argmax(hypothesis, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(y_test, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy = accuracy_score(hypothesis)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())


    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x1:x_train, y1:y_train})

        if step %200 == 0:
            print(step, loss_val)

    
    predict = sess.run(hypothesis, feed_dict={x1:x_test})
    print('\n',predict, sess.run(tf.math.argmax(predict,1)))
    acc1 = sess.run(tf.math.argmax(predict,1))
    
    
'''
predicted = tf.arg_max(hypo,1)

acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))
'''