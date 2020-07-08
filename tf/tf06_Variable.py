import tensorflow as tf

tf.compat.v1.set_random_seed(66) # 변수값 랜덤값으로 시작위치 결정 (없어도 됨)


w = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


w = tf.Variable([0.3], tf.float32)

sess = tf.Session() # 메모리 열어서 작업
sess.run(tf.global_variables_initializer())
aaa = sess.run(w)
print(aaa)
sess.close()
# 메모리를 열었으니 닫는 작업



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = w.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session = sess)
print(ccc)
sess.close()










