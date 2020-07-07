import tensorflow as tf

tf.compat.v1.set_random_seed(66) # 변수값 랜덤값으로 시작위치 결정 (없어도 됨)

x_train = [1,2,3]
y_train = [3,5,7]

w = tf.Variable(tf.random.normal([1]), name = 'weight') # dimention 즉 차원에대해서 1-D로 지정
b = tf.Variable(tf.random.normal([1]), name = 'bias') 
# 초기화가 필요함


sess = tf.compat.v1.Session()

# sess.run(tf.compat.v1.global_variables_initializer()) # 초기화
# print(sess.run(w))


hypothesis = x_train*w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse : mean squared error
# reduces all dimensions 전체평균을 구한다
# 모든 차원이 제거되고 단 하나의 스칼라 값이 출력된다



train = tf.train.GradientDescentOptimizer(learning_rate=  0.01).minimize(cost)
# optimizer => 최족화 경사하강법 adam

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b])
        '''
        train은 따로 출력 안하고 cost => cost_val로     w => w_val로       b => b_val로 출력
        train은 하지만 출력만 안하는거임
        '''

        if step % 20 == 0: # 20번마다 출력
            print(step, cost_val, w_val, b_val)
'''
1920 1.7834054e-05 [1.004893] [-0.01112291]
1940 1.6197106e-05 [1.0046631] [-0.01060018]
1960 1.4711059e-05 [1.004444] [-0.01010205]
1980 1.3360998e-05 [1.0042351] [-0.00962736]
2000 1.21343355e-05 [1.0040361] [-0.00917497]

w와 b값들이 수렴하고 있는 것으로 보임
'''

'''
c언어에서는 변수나 배열에서 초기화를 해줘야 값이 들어가게 되는데 안에 쓰레기값이 들어가 있기 때문입니다. 
예를 들어 스테인리스 그릇의 경우 한번 세척후 사용하는것과 비슷하다고 생각하는게 편할듯
'''