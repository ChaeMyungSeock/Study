from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
data_generator = ImageDataGenerator(rescale=1./255)

# 이미지 데이터 픽셀 값은 0~255범위 값을 가짐 0~1사이로 정규화
# 서브 디렉토리의 폴더명이 해당 폴더에 들어있는 이미지들의 라벨이 됨

train_generator = data_generator.flow_from_directory(
    './mini_proj/',
    target_size=(100,100),
    batch_size=1,
    class_mode='binary')
print(len(train_generator))

x_train, y_train = train_generator.next()
# print(x_train)
# print(y_train.shape)
x_train = x_train.reshape(100,100,3)
# plt.imshow(x_train[0])
# plt.show()

# def fourier():
#     img = x_train
#     b,g,r = cv2.split(img)
#     img = cv2.merge([r,g,b])
#     img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift)) # 스펙트럼 구하는 식

#     rows,cols = img.shape
    
#     crow, ccol = int(rows/2), int(cols/2)

#     d = 10
#     fshift[crow-d:crow+d, ccol-d:ccol+d] = 1

#     #푸리에 변환결과를 다시 이미지로 변환
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.abs(img_back)

   
#     # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
#     # # 주파수 영역의 이미지 정중앙의 60x60 크기 영역에 있는 값
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.abs(img_back)

#     #threshold를 적용하기 위해 float type을 int type으로 변환
#     img_new = np.uint8(img_back);
#     ret, thresh = cv2.threshold(img_new,30,255,cv2.THRESH_BINARY_INV)

#     plt.subplot(131), plt.imshow(img, cmap='gray')
#     plt.title('Original image'), plt.xticks([]), plt.yticks([])
    
#     plt.subplot(132), plt.imshow(img, cmap='gray')
#     plt.title('After HPF'), plt.xticks([]), plt.yticks([])

#     plt.subplot(133), plt.imshow(img, cmap='gray')
#     plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

#     plt.show() 

# fourier()
