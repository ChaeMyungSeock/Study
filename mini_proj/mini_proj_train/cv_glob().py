from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
import cv2.cv2 as cv2
import time
import os
from PIL import Image


# glob() cv로 파일안에 확장명 이미지 파일을 모두 불어옴(푸리에 변환을 위해서임)

path = glob.glob("D:/Study/IU/newtest/*.jpg")
path1 = "D:/Study/mini_proj_train2/img/"
print(path.__class__)
cv_img = []

# for img in path:
#     n = cv.imread(img)
#     cv_img.append(n)
# cv_img = np.array(cv_img)
''' 
이미지에 적용이 되어 중심이 저주파 영역, 주변이 고주파 영역을 나타냄.
푸리에 변환을 하여 저주파 또는 고주파를 제거하고 다시 역으로 이미지로 변환 함으로써 이미지가공을 할 수 있음.
(ex; 푸리에 변환 후 중심의 저주파를 제거하고 다시 Image로 전환 하면 이미지의 경계선만 남게 됨.
푸리에 변환 후 주변의 고주파를 제거하면 모아레 패턴(휴대폰으로 모니터를 찍었을 때 나타나는 현상)
을 제거할 수 있음.(모니터의 고주파를 제거함.)
'''
i=0

for image_title in path:
    img = cv.imread(image_title)
    b,g,r = cv.split(img)
    img = cv.merge([r,g,b])
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Fourier Transfrom을 적용
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift)) # 스펙트럼 구하는 식

    rows, cols = img.shape
    crow, ccol = rows//2, cols//2 # 이미지의 중심좌표
    
    # 중앙에서 10x10 사이즈의 사각형의 값을 1로 설정함. 중앙의 저주파를 모두 제거
    # 저주파를 제거하였기 때문에 배경이 사라지고 경계선만 남게 됨.
    d = 10

    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    img_and_magnitude = np.concatenate((img, magnitude_spectrum), axis=1)
    fshift[crow-d:crow+d, ccol-d:ccol+d] = 1

    #푸리에 변환결과를 다시 이미지로 변환
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # cv.imshow(image_title, img_and_magnitude)
    cv_img.append(img_back)
    cv.imwrite(os.path.join(path1,f"pic{i}.jpg"), img_back)
    i = i+1
    print(i)
    # img_back = np.array(img_back)


    # cv.SaveImage('pic'+str(i)+'.jpg', format(i), img)
    # #THRESHOLD를 적용하기 위해 float type를 int type으로 변환
    # img_new = np.uint8(img_back);
    # ret, thresh = cv.threshold(img_new,30,255,cv.THRESH_BINARY_INV)
    # image = Image.fromarray(cv_img)
    

# cv.waitKey(0)
# cv.destroyAllWindows()

cv_img = np.array(cv_img)
print(cv_img[1])
# print(cv_img)

