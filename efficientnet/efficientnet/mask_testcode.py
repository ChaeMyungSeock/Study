#from efficientnet import model
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Flatten,MaxPool2D, Dropout,Dense,Conv2D, GlobalAveragePooling2D,Input
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from efficientnet.tfkeras import EfficientNetB0
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import glob
from efficientnet import preprocessing
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import dlib
import cv2
# import openface
#from efficientnet import weights
from keras import backend as K

# 과제 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

x = np.load('/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/label/test.npy')/255.0
y = np.loadtxt('/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/label/with.txt')

x = np.array(x)
x = x/255.0
# print(x)
print(x.shape)
# #
# # print(x[1].shape)
# # 2. 모델
model = load_model('/home/john/Study/mask/realmask_model_10_06.h5')
# model.save('/home/john/Study/mask/mask_model.h5')
model.compile(metrics = ['acc'])
acc = model.evaluate(x,)
#
y_predict = model.predict(x)
print(y_predict)
# print("loss : ", loss)
# print("acc : ", acc)

# for i in range(len(x)):

for i in range(len(x)):

    result = np.where(y_predict[i] > 0.5, 1, 0)
    file = files1[i]
    img1 = cv2.imread(file)
    tit = "%.7f"%y_predict[i]21
    re = str(result)
    cv2.imshow(re, img1)
    cv2.waitKey(0)  # key if number msecond
    cv2.destroyWindow(tit)
    if result==1:
        print("mask_within")
    else :
        print("mask_without")

# a = round(y_predict[i])

# print(y_predict)
# print('f1', metrics.f1_score(y,p) )
#  y=>true p => predict
#
# # print('진짜 끝')
#
# # error
# # 'JpegImageFile' object has no attribute 'ndim' # numpy dimension error

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
