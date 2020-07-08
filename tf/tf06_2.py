import tensorflow as tf

tf.compat.v1.set_random_seed(66) # 변수값 랜덤값으로 시작위치 결정 (없어도 됨)

x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_test_data = [4]
w = tf.Variable(tf.random.normal([1]), name = 'weight') # dimention 즉 차원에대해서 1-D로 지정
b = tf.Variable(tf.random.normal([1]), name = 'bias') 
# 초기화가 필요함

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])





sess = tf.compat.v1.Session()

# sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
# print(sess.run(w))


hypothesis = x_train*w + b
# feed_dict = {x : x_train, y : y_train}

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse : mean squared error
# reduces all dimensions 전체평균을 구한다
# 모든 차원이 제거되고 단 하나의 스칼라 값이 출력된다



train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.12).minimize(cost)
# optimizer => 최족화 경사하강법 adam

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict={x_train :x_train_data, y_train : y_train_data})
        '''
        train은 그냥 optimizer역활
        feed_dict에서는 우리가 모델링을 먼저 해놓고 정제해놓은 데이터를 불러옴
        hypotheis는 mse를 구현하기위한 수단

        '''

        if step % 20 == 0: # 20번마다 출력
            print(step, cost_val, w_val, b_val)

        # print(step, cost_val, w_val, b_val)

    # predict해보자
    print("예측 : ", sess.run(hypothesis, feed_dict ={x_train:x_test_data}))


'''
lr = 0.1 step = 540 /////// lr 0.12 step = 460

lr = 0.001 step = 20000 1.1022381e-05 [1.9961436] [1.0087627] 아모르직다 수렴중

lr = 1 step = 20 inf [1.7777886e+21] [7.8205226e+20] 무한대로 뻗침(error가 cost가)
'''