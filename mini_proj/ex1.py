from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import tensorflow as tf
caltech_dir = 'D:/study/efficientnet/data/test/VOCdevkit/VOC2012/JPEGImages'


image_w = 224
image_h = 224

print('시작')
pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)


X = np.array(X)
X = X.astype(float) / 255
print(X.shape)

np.save('D:/study/data/test',X)
print('끝났습니다')