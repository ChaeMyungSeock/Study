from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten,ZeroPadding2D,Conv2DTranspose,UpSampling2D
# from keras.layers.convolutional import UpSampling2D
import numpy as np
def autoencoder(hidden_layer_size, x_train):
    print(x_train.shape)
    model = Sequential()
    model.add(Conv2D(filters= hidden_layer_size*16, padding = 'valid',kernel_size = (3,3),input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),  activation='relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filters= hidden_layer_size*4, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filters= hidden_layer_size*2, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= 32, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(UpSampling2D(size = (2,2),interpolation='nearest',data_format=None))
    model.add(Conv2DTranspose(filters= 16, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= 8, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(UpSampling2D(size = (2,2),interpolation='nearest',data_format=None))
    model.add(Conv2DTranspose(filters= 3, padding = 'valid',kernel_size = (1,1),  activation='relu'))


    model.summary()
    return model

from tensorflow.keras.datasets import cifar10
train_set, test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape)


# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] , x_train.shape[2],1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] , x_test.shape[2],1)
x_train = x_train / 255.
x_test = x_test / 255.


x_train_noised = x_train + np.random.normal(0,1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0,1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)



model = autoencoder(hidden_layer_size=32, x_train=x_train)


# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc']) # loss 0.0102
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093

model.fit(x_train_noised, x_train, epochs=20)

output = model.predict(x_test)
output1 = model.predict(x_test_noised)


from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15),(ax16, ax17, ax18, ax19, ax20)) = plt.subplots(4,5, figsize = (20,7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32,32,3))
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출려한 이미지를 아래 그린다.

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(32,32,3))
    if i == 0:
        ax.set_ylabel("INPUT_NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(32,32,3))
    if i == 0:
        ax.set_ylabel("INPUT_NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


for i, ax in enumerate([ax16, ax17, ax18, ax19, ax20]):
    ax.imshow(output1[random_images[i]].reshape(32,32,3))
    if i == 0:
        ax.set_ylabel("INPUT_NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
