from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import cv2 as cv
# caltech_dir = '/data/mask/data/within_mask'
# caltech_dir1 = '/data/mask/data/without_mask'
# test_dir = '/data/mask/test/test'
caltech_dir = '/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/withmask'
caltech_dir1 = '/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/withoutmask'


files = glob.glob(caltech_dir+"/*.*")
files1 = glob.glob(caltech_dir1+"/*.*")

# files = glob.glob(caltech_dir+"/*.*")
# files1 = glob.glob(test_dir+"/*.*")

x=[]
for img in files:
    img = Image.open(img)
    img = img.convert("RGB")
    data = np.asarray(img)
    x.append(data)
# print('1')
for img1 in files1:
    img1 = Image.open(img1)
    img1 = img1.convert("RGB")
    data = np.asarray(img1)
    x.append(data)
print('0')

x = np.array(x)
# x = x.astype(float) / 255
print(x.shape)

np.save('/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/label/test.npy',x)

# np.save('/home/john/Study/efficientnet/data/mask_test_0',x)
# (282, 224, 224, 3)
print('끝났습니다')
