import numpy as np
import glob
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img



train_data = ImageDataGenerator(
                                    rotation_range =40,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip = True,
                                    fill_mode='nearest')

# test_data = ImageDataGenerator(rescale=1./255)
for i in range(52,62):
    
    img = load_img('D:/Study/Hexapod_Bot/image/data/inner_road_test/'+str(i)+'.jpg')  # PIL 이미지
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)  # (1, 360, 360, 3) 크기의 NumPy 배열

    # 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
    # 지정된 폴더에 저장합니다.
    j = 0
    for batch in train_data.flow(x, batch_size=1,
                            save_to_dir='./Hexapod_Bot/image/data/preview_test_1', save_prefix=str(j), save_format='jpeg'):
        j += 1
        if j > 20:
            break  # 이미지를 j장까지 생성하고 마칩니다


# train 20
# test 10