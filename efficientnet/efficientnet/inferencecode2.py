# from efficientnet import model
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, MaxPool2D, Dropout, Dense, Conv2D, GlobalAveragePooling2D, Input
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
import matplotlib.patches as patches
sp = dlib.shape_predictor('/home/john/Study/mask/landmark/shape_predictor_5_face_landmarks.dat')



# import openface
# from efficientnet import weights
# 과제 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

caltech_dir1 = '/data/mask/test/test_mask'
# data log

files1 = glob.glob(caltech_dir1 + "/*.*")

x = []

for img in files1:
    face_detector = dlib.get_frontal_face_detector()
    img_det = dlib.load_rgb_image(img)
    dets = face_detector(img_det, 1)
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')

    objs = dlib.full_object_detections()

    for detection in dets:
        s = sp(img_det, detection)
        objs.append(s)
        for point in s.parts():
            circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
    face = dlib.get_face_chips(img_det, objs, size =224, padding=0.2)

    plt.imshow(face[0])
    plt.show()

#     img1 = cv2.imread(face)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     plt.imshow(img1)
#     plt.show()
#     img = np.asarray(img1)
#     print(img.shape)
#
#
x = np.array(x)
x = x / 255.0
print(x)


# # 2. 모델
# model = load_model('/home/john/Study/mask/mask_model.h5')

#
# y_predict = model.predict(x)
# print(y_predict)
# # print("loss : ", loss)
# # print("acc : ", acc)
#
# # for i in range(len(x)):
# for i in range(len(x)):
#     result = np.where(y_predict[i] > 0.5, 1, 0)
#     if result == 1:
#         print("mask_within")
#     else:
#         print("mask_without")
#
