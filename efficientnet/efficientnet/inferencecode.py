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
# 과제 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# caltech_dir1 = '/data/mask/test_mask'
caltech_dir1 = '/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/withmask'

files1 = glob.glob(caltech_dir1+"/*.*")

x=[]
print(len(files1))
for img in files1:
    # face_detector = dlib.get_frontal_face_detector()

    img1 = cv2.imread(img)
    # face = face_detector(img1)


    # cv2.rectangle(img1, (face[0].left(), face[0].top()), (face[0].right(), face[0].bottom()), (0,0,255),2)



    # win = dlib.image_window()
    # win.set_image(img1)
    # win.add_overlay(face)
    # img1 = img1[ face[0].top():face[0].bottom(),face[0].left():face[0].right()]
    # img1 = cv2.resize(img1, dsize=(224,224), interpolation=cv2.INTER_AREA)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img = np.asarray(img1)
    # print(img.shape)

    x.append(img)



x = np.array(x)
x = x/255.0
# print(x)
print(x.shape)
# #
# # print(x[1].shape)
# # 2. 모델
model = load_model('/home/john/Study/mask/realmask_model_10_06.h5')
# model.save('/home/john/Study/mask/mask_model.h5')

#
y_predict = model.predict(x)
print(y_predict)
# print("loss : ", loss)
# print("acc : ", acc)

# for i in range(len(x)):

for i in range(len(x)):

    # result = np.where(y_predict[i] > 0.5, 1, 0)
    file = files1[i]
    img1 = cv2.imread(file)
    y_predict[i] *= 100
    tit = "%.7f"%y_predict[i]
    # re = str(result)
    cv2.imshow(tit, img1)
    cv2.waitKey(0)  # key if number msecond
    cv2.destroyWindow(tit)
    # if result==1:
    #     print("mask_within")
    # else :
    #     print("mask_without")

# a = round(y_predict[i])

# print(y_predict)
# print('f1', metrics.f1_score(y,p) )
#  y=>true p => predict
#
# # print('진짜 끝')
#
# # error
# # 'JpegImageFile' object has no attribute 'ndim' # numpy dimension error