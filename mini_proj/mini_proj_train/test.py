import os
import cv2
import numpy as np
from sklearn.utils import shuffle

images = []
labels = []
directory = 'D:/Study/mini_proj_train/images/'

for label, names in enumerate(os.listdir(directory)):
    try :
        for image_file in os.listdir(directory + names):
            image = cv2.imread(directory +  names + r'/' + image_file)
            image = cv2.resize(image, (100,100))
            images.append(image)
            labels.append(label)
    
    except Exception as e:
        print(str(e))

shuffle(images, labels, random_state=5)

lmages = np.array(images)
labels = np.array(labels)

fime_name = 'train_data_ex' # change you want name
Save = np.savez(directory+fime_name, x=images, y=labels)

