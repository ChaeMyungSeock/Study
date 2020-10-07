from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import matplotlib.pyplot as plt

# path
caltech_dir1 = '/home/ubuntu/data/mask/test/within_mask'
caltech_dir2 = '/home/ubuntu/data/mask/test/without_mask'

# dataloder
files1 = glob.glob(caltech_dir1+"/*.*")
files2 = glob.glob(caltech_dir2+"/*.*")

# labelfile create
# label1 = open('/home/ubuntu/data/mask/label/mask_test_label.txt', 'w')

# label data insert
label1 = open('/home/ubuntu/data/mask/label/mask_test_label.txt', 'a')

print(len(files1))
print(len(files2))

for i in files1:
    label1.write('1\n')

for i in files2:
    label1.write('0\n')

#



#
# # (300, 224, 224, 3) => mask on => 1
# # (282, 224, 224, 3) => mask off => 0
#
# for i in range(282):
#     label.write('0\n')
#
#
#

label1.close()

