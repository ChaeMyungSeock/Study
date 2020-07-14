import tensorflow as tf
import numpy as np

# hihello

idx2char = ['e','h','i','l','o']

_data = np.array([['h','i','h','e','l','l','o']]).reshape(-1,1)
print(_data.shape)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()


print("=================")
print(_data)
print(type(_data))
print(_data.dtype)


x_data = _data[:6,] # hihell
y_data = _data[1:,] # ihello

print("========== x =============")
print(x_data)
print("========== y =============")
print(y_data)

print("========== y argmax =============")
y_data = np.argmax(y_data, axis=1)
print(y_data)
print(y_data.shape) # (6,)

print(x_data.shape)

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)


print(x_data.shape)
print(y_data.shape) # 6

output = 100
batch_size = 1 # 전체 행

X = tf.compat.v1.placeholder(tf.float32, shape = (None,x_data[-1].shape[0],x_data[-1].shape[-1]), name = 'data_X')
Y = tf.compat.v1.placeholder(tf.int32, shape = (None,y_data[-1].shape[0]), name = 'data_Y')

print(X)
print(Y)


# 2. 모델 구성
# model.add(LSTM(output, input_shape = (6,5)))
cell = tf.keras.layers.LSTMCell(output)
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
print(hypothesis) # shape=(?, 6, 100)


# 3-1. 컴파일
weights = tf.ones([batch_size,x_data[-1].shape[0]])
# weights 선형을 default로 잡고 가것다

print(weights)
'''

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights
)

cost = tf.reduce_mean(sequence_loss)

train  = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
prediction = tf.argmax(hypothesis, axis=2)

# axis = 0 행 axis = 1 열 axis = 2 피쳐


# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X : x_data, Y : y_data})
        result = sess.run(prediction, feed_dict={X : x_data})
        print(i, "loss : ", loss , "prediction : ", result, "true Y : ", y_data)

    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("\n prediction str : ", ''.join(result_str))
        
        '''