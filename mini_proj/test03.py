from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import tensorflow as tf
caltech_dir = 'D:/Study/mini_proj_testdata/img'


image_w = 36
image_h = 36

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
X = X.reshape(X.shape[0], 36, 36*3)
model = load_model('./model/IU_Classify.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0
print(prediction)

for i in prediction:
    if i >= 0.5: print("해당 "  +str(cnt+1) +   " 번째 이미지는 IU 로 추정됩니다.")
    else : print("해당 " + str(cnt+1) +  "이미지는 태연으로 추정됩니다.")
    cnt += 1