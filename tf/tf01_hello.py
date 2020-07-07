import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World")
# constant 고정된 값 변하지 않음

print(hello)
# Tensor("Const:0", shape=(), dtype=string)
# 텐서플로우에서는 텐서플로우 방식으로 값을 반환

sess = tf.Session()
print(sess.run(hello))
'''
텐서플로우에서 값을 확인하기 위해서는 텐서플로우 형식의 값을 Session을 통해 우리가 편한 값으로 보여줌 
'''