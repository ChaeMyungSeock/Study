from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import cv2 as cv
caltech_dir1 = '/data/mask/new'
caltech_dir2 = '/data/mask/new2'
caltech_dir3 = '/data/mask/new3'
caltech_dir4 = '/data/mask/data/without_mask'


files = glob.glob(caltech_dir1+"/*.*")
files1 = glob.glob(caltech_dir2+"/*.*")
files2 = glob.glob(caltech_dir3+"/*.*")
files3 = glob.glob(caltech_dir4+"/*.*")

x=[]
for img in files:
    img = Image.open(img)
    img = img.convert("RGB")
    data = np.asarray(img)
    x.append(data)
x = x[-300:]
for img1 in files1:
    img1 = Image.open(img1)
    img1 = img1.convert("RGB")
    data = np.asarray(img1)
    x.append(data)
x = x[:300] + x[-300:]

for img2 in files2:
    img2 = Image.open(img2)
    img2 = img2.convert("RGB")
    data = np.asarray(img2)
    x.append(data)
x = x[:600] + x[-200:]

for img2 in files3:
    img2 = Image.open(img2)
    img2 = img2.convert("RGB")
    data = np.asarray(img2)
    x.append(data)
x = x[:800] + x[-800:]
x = np.array(x)
print(x.shape)

#
np.save('/home/john/Study/mask/train_data',x)
print('끝났습니다')
