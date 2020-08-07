import glob
import numpy as np
import os
import cv2
from os import rename, listdir
from PIL import Image
import os
# 현재 위치의 파일 목록
file_path = './Hexapod_Bot/image/data/preview_train_1/'
files = os.listdir(file_path)

print(len(files))
# 이미지들 픽셀데이터화 시키고 넘파이로 저장
images = []
for img in range(len(files)):

    jpg = cv2.imread('./Hexapod_Bot/image/data/preview_train_1/' + str(img)+'.jpg')
    images.append(jpg)

print(images)
images = np.array(images)
print(images.shape)
np.save('./Hexapod_Bot/image/data/use_data/train/train_1',images)