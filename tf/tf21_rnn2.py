import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape) # (10,)


# RNN 모델을 짜시오!
def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    return np.array(aaa)

size = 6 # 앞에 5개 x 뒤에 하나 y

data = split_x(dataset,6)
print(data.shape)

x_data = data[:,:5]
y_data = data[:,5]
print(x_data.shape)
print(y_data.shape)

x_data = x_data.reshape(5,5,1)
y_data = y_data.reshape(5,1)

X = tf.compat.v1.placeholder(tf.float32, shape = (None,5,1), name = 'data_X')
Y = tf.compat.v1.placeholder(tf.int32, shape = (None,1), name = 'data_Y')

print(X) # 5,1
print('Y : ', Y) # 1


output = 10
batch_size = 5 # 전체 행

rnn = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(rnn, X, dtype = tf.float32)
print(hypothesis) # shape=(?, 5, 10)


weights = tf.ones([batch_size,1])
print(weights)


sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights
)

cost = tf.reduce_mean(sequence_loss)

train  = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
prediction = tf.argmax(hypothesis, axis=2)

# print(prediction)
# axis = 0 행 axis = 1 열 axis = 2 피쳐


# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X : x_data, Y : y_data})
        result = sess.run(prediction, feed_dict={X : x_data})
        print(i, "loss : ", loss , "prediction : ", result, "true Y : ", y_data)

