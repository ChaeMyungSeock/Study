from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree


caltech_dir = 'D:/study/efficientnet/data/train_val/VOCdevkit/VOC2012/Annotations'



print('시작')
label = open('D:/study/data/train_val_label.txt', "a")
files = glob.glob(caltech_dir+"/*.xml")
for i, f in enumerate(files):
    doc = ET.parse(f)
    root = doc.getroot()
    size_tag = root.findall("object")
    if size_tag[0].find("name").text == 'person':
        label.write('0\n')
    elif size_tag[0].find("name").text == 'aeroplane':
        label.write('1\n')
    elif size_tag[0].find("name").text == 'tvmonitor':
        label.write('2\n')
    elif size_tag[0].find("name").text == 'train':
        label.write('3\n')
    elif size_tag[0].find("name").text == 'boat':
        label.write('4\n')
    elif size_tag[0].find("name").text == 'dog':
        label.write('5\n')
    elif size_tag[0].find("name").text == 'bird':
        label.write('6\n')
    elif size_tag[0].find("name").text == 'bicycle':
        label.write('7\n')
    elif size_tag[0].find("name").text == 'bottle':
        label.write('8\n')
    elif size_tag[0].find("name").text == 'sheep':
        label.write('9\n')
    elif size_tag[0].find("name").text == 'diningtable':
        label.write('10\n')
    elif size_tag[0].find("name").text == 'horse':
        label.write('11\n')
    elif size_tag[0].find("name").text == 'motorbike':
        label.write('12\n')
    elif size_tag[0].find("name").text == 'sofa':
        label.write('13\n')
    elif size_tag[0].find("name").text == 'cow':
        label.write('14\n')
    elif size_tag[0].find("name").text == 'cat':
        label.write('15\n')
    elif size_tag[0].find("name").text == 'bus':
        label.write('16\n')
    elif size_tag[0].find("name").text == 'pottedplant':
        label.write('17\n')
    elif size_tag[0].find("name").text == 'chair':
        label.write('18\n')
    elif size_tag[0].find("name").text == 'car':
        label.write('19\n')
print('끝났습니다')