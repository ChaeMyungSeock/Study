

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten,Conv2DTranspose


# def autoencoder(hidden_layer_size):

    # model = Sequential()
    # model.add(Conv2D(filters= hidden_layer_size*10, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    # model.add(ZeroPadding2D(padding = (1,1)))
    # model.add(Conv2D(filters= hidden_layer_size*8, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    # model.add(ZeroPadding2D(padding = (1,1)))
    # model.add(Conv2D(filters= hidden_layer_size*6, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    # model.add(ZeroPadding2D(padding = (1,1)))
    # model.add(Conv2D(filters= hidden_layer_size*4, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    # model.add(ZeroPadding2D(padding = (1,1)))
    # model.add(Conv2D(filters= hidden_layer_size*2, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    # model.add(ZeroPadding2D(padding = (1,1)))
    # model.add(Conv2D(filters= 1, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='sigmoid'))
    # model.add(ZeroPadding2D(padding = (1,1)))
#     return model

def autoencoder(hidden_layer_size):

    model = Sequential()
    model.add(Conv2D(filters= hidden_layer_size*32, padding = 'valid',kernel_size = (3,3),input_shape=(28,28,1),  activation='relu'))
    model.add(Conv2DTranspose(filters= hidden_layer_size*16, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2D(filters= hidden_layer_size*8, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= hidden_layer_size*4, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2D(filters= hidden_layer_size*2, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= hidden_layer_size, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2D(filters= hidden_layer_size, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= hidden_layer_size, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2D(filters= hidden_layer_size//2, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= hidden_layer_size//4, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2D(filters= hidden_layer_size//8, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.add(Conv2DTranspose(filters= 1, padding = 'valid',kernel_size = (3,3),  activation='relu'))
    model.summary()
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
x_train = x_train / 255.
x_test = x_test / 255.

model = autoencoder(hidden_layer_size=32)

# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc']) # loss 0.0102
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 0.093

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5, figsize = (20,7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출려한 이미지를 아래 그린다.

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()