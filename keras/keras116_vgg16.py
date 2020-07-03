from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.applications import VGG16,VGG19, Xception, ResNet101,ResNet101V2,ResNet152
from keras.applications import ResNet152V2,ResNet50,ResNet50V2,InceptionV3,InceptionResNetV2
from keras.applications import MobileNet,MobileNetV2,DenseNet121,DenseNet169,DenseNet201
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation,Flatten
from keras.applications.nasnet import NASNetLarge, NASNetMobile

# applications = [VGG19, Xception, ResNet101, ResNet101V2, ResNet152,ResNet152V2, ResNet50, 
#                 ResNet50V2, InceptionV3, InceptionResNetV2,MobileNet, MobileNetV2, 
#                 DenseNet121, DenseNet169, DenseNet201]



# take_model = VGG16() # (None, 224, 224,3)
# vgg16 = VGG16() # (None, 224, 224,3)

# model.summary()

# model = Sequential()
# model.add(vgg16(input_tensor=True))
# model.add(Dense(256, name = 'hidden1'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10,name = 'output',activation='softmax'))

# model.summary()