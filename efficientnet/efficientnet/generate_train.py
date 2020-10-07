#
#from efficientnet import model
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Flatten,MaxPool2D, Dropout,Dense,Conv2D, GlobalAveragePooling2D,Input
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from efficientnet.tfkeras import EfficientNetB0,EfficientNetB1
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#from efficientnet import weights
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 과제 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/data/mask/generate/generate_trainmask',
    target_size=(224,224),
    batch_size=2,
    class_mode = 'binary'
)
test_generator = test_datagen.flow_from_directory(
    '/data/mask/generate/generate_trainmask',
    target_size=(224,224),
    batch_size=2,
    class_mode='binary'
)



# 2. 모델
model = Sequential()
model.add(EfficientNetB1(include_top=False,pooling = 'avg'))
model.add(Dropout(0.1, name='hidden1'))
#model.add(GlobalAveragePooling2D(name='hidden2'))
model.add(Dense(1, activation='sigmoid',name='s1'))
model.summary()
# 3. 훈련


# tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch')

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
hist = model.fit_generator(train_generator,
                           steps_per_epoch=15,
                           epochs=20,
                           validation_data=test_generator,
                           validation_steps=5)
# hist = model.fit(x_train, y_train, batch_size=256, epochs=10,validation_split=0.2)

model.save('/home/john/Study/mask/mask_model_0925.h5')



# 4. 평가, 예측

scores = model.evaluate_generator(test_generator, steps=5)
print(scores)

# #
# print('f1', metrics.f1_score(y_test,y_predict) )
# print(_f1_score)
# # print(y_predict)

# # print('진짜 끝')
#
import matplotlib.pyplot as plt

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train acc','val acc', 'train loss', 'val loss'])

plt.show()