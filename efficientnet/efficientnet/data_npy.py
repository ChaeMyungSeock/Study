from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import cv2 as cv


# train_data path
# caltech_dir1 = '/home/ubuntu/data/mask/train/within_mask'
# caltech_dir2 = '/home/ubuntu/data/mask/train/without_mask'

# test_data path
caltech_dir1 = '/home/ubuntu/data/mask/test/within_mask'
caltech_dir2 = '/home/ubuntu/data/mask/test/without_mask'


# img glob
files1 = glob.glob(caltech_dir1+"/*.*")
files2 = glob.glob(caltech_dir2+"/*.*")


x=[]
for img in files1:
    img = Image.open(img)
    img = img.convert("RGB")
    data = np.asarray(img)
    x.append(data)
print('1')

for img1 in files2:
    img1 = Image.open(img1)
    img1 = img1.convert("RGB")
    data = np.asarray(img1)
    x.append(data)
print('0')

# numpy
x = np.array(x)
print(x.shape)

# npy save
np.save('/home/ubuntu/data/mask/label/mask_test_data.npy',x)

print('끝났습니다')


# train
# (4028, 224, 224, 3)

# test
# (198, 224, 224, 3)

