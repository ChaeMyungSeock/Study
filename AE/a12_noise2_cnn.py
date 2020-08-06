from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten,ZeroPadding2D,Conv2DTranspose,UpSampling2D
# from keras.layers.convolutional import UpSampling2D
import numpy as np
def autoencoder(hidden_layer_size):

    model = Sequential()
    model.add(Conv2D(filters= hidden_layer_size*16, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filters= hidden_layer_size*4, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filters= hidden_layer_size, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    model.add(Conv2DTranspose(filters= 16, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(UpSampling2D(size = (2,2),interpolation='nearest',data_format=None))
    model.add(Conv2DTranspose(filters= 8, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= 4, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(UpSampling2D(size = (2,2),interpolation='nearest',data_format=None))
    model.add(Conv2DTranspose(filters= 1, padding = 'valid',kernel_size = (1,1),  activation='relu'))


    model.summary()
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] , x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] , x_test.shape[2],1)
x_train = x_train / 255.
x_test = x_test / 255.

# 노이즈 추가
x_train_noised = x_train + np.random.normal(0,0.5, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.5, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


'''
평균을 기준으로 표준편차만큼 랜덤하게 값을 부여 ( 0 => 평균     0.5 => 표준편차)
0~1로 데이터를 축소 시켜 놓았는데 평균 0, 0.5 표준편차로 하게 되면 음수 양수를 가지고 값이 분포하는데
여기서 값이 0~1의 scaling값으로 고정 시키기 위해서
clip => 0이하의 숫자를 0으로 채움
'''



model = autoencoder(hidden_layer_size=32)


# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc']) # loss 0.0102
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), 
        (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5, figsize = (20,7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출려한 이미지를 아래 그린다.

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("INPUT_NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("OUTPUT_NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
