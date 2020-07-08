# hypothesis를 구하시오
# H = Wx + b

import tensorflow as tf
import matplotlib
tf.compat.v1.set_random_seed(777) # 변수값 랜덤값으로 시작위치 결정 (없어도 됨)

x = [1,2,3]
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable([0.3], dtype = tf.float32)
b = tf.Variable([1], dtype = tf.float32)

hypersis = w*x + b


sess = tf.Session() # 메모리 열어서 작업
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypersis)
print('hypothesis : ', aaa) 
sess.close()
# 메모리를 열었으니 닫는 작업



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypersis.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypersis.eval(session = sess)
print(ccc)
sess.close()


