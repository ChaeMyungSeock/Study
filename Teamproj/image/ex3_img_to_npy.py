import numpy as np
from PIL import Image
import os

# path_dir = './Hexapod_Bot/image/data/preview_train_0/'
# file_list = os.listdir(path_dir)

# for jpg in file_list:
#     image = Image.open(path_dir + jpg)
#     pixel = np.array(image)
#     jgp = jpg.split('.')[0]
#     np.save("./Hexapod_Bot/image/data/use_data/preview_train_0_data/"+jpg,pixel)


# path_dir = './Hexapod_Bot/image/data/preview_train_1/'
# file_list = os.listdir(path_dir)

# for jpg in file_list:
#     image = Image.open(path_dir + jpg)
#     pixel = np.array(image)
#     jgp = jpg.split('.')[0]
#     np.save("./Hexapod_Bot/image/data/use_data/preview_train_1_data/"+jpg,pixel)
    
# path_dir = './Hexapod_Bot/image/data/preview_test_0/'
# file_list = os.listdir(path_dir)

# for jpg in file_list:
#     image = Image.open(path_dir + jpg)
#     pixel = np.array(image)
#     jgp = jpg.split('.')[0]
#     np.save("./Hexapod_Bot/image/data/use_data/preview_test_0_data/"+jpg,pixel)

path_dir = './Hexapod_Bot/image/data/preview_test_1/'
file_list = os.listdir(path_dir)

for jpg in file_list:
    image = Image.open(path_dir + jpg)
    pixel = np.array(image)
    jgp = jpg.split('.')[0]
    np.save("./Hexapod_Bot/image/data/use_data/preview_test_1_data/"+jpg,pixel)
    