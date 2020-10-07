from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import matplotlib.pyplot as plt

# caltech_dir = '/data/mask/data/within_mask'
# caltech_dir1 = '/data/mask/data/without_mask'
caltech_dir1 = '/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/withoutmask'

files = glob.glob(caltech_dir1+"/*.*")
# #
# label = open('/home/john/Study/efficientnet/data/label.txt', 'a')

# label = open('/home/john/Study/pytorch_retinaface/facedetect/target_mask/train_label.txt', 'a')
label = open('/home/john/Study/pytorch_retinaface/sg_maskdata/test/data/label/with.txt', 'a')
# #
#
print(len(files))
for i in files:
    label.write('1\n')
# #
#
#
# label.close()

# label = open('/home/john/Study/mask/test_label.txt', 'a')
#
# # (300, 224, 224, 3) => mask on => 1
# # (282, 224, 224, 3) => mask off => 0
#
# for i in range(282):
#     label.write('0\n')
#
#
#
label.close()
