# 3 + 4 + 5
# 4 - 3
# 3*4
# 4/2

import tensorflow as tf
node0 = tf.constant(2.0, tf.float32)
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.constant(5.0, tf.float32)

node_add = tf.add(tf.add(node1,node2),node3)
node_sub = tf.subtract(node2, node1)
node_multi = tf.multiply(node1, node2)
node_div = tf.divide(node2, 2)

sess = tf.Session()
# print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node_add) : ", sess.run(node_add))
print("sess.run(node_sub) : ", sess.run(node_sub))
print("sess.run(node_multi) : ", sess.run(node_multi))
print("sess.run(node_div) : ", sess.run(node_div))

# node3 = tf.add(node1, node2)