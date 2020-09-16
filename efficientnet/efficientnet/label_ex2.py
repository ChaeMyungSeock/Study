from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree


# caltech_dir = 'D:/data/voc/VOC2012trainval/VOCdevkit/VOC2012/Annotations'
#
for i in range(10):
    label = open('D:/data/voc/YOLOv4_labele/test' + i + ".txt", "w")
    label.close()

#
# caltech_dir = 'D:/data/voc/VOC2012trainval/VOCdevkit/VOC2012/Annotations/2007_000364'
#
# files = glob.glob(caltech_dir+"/*.xml")
#
# print('시작')
# for i in range(len(files)):
#     label = open('D:/data/voc/YOLOv4_labele/test'+i+".txt", "a")
#     # files = glob.glob(caltech_dir+"/*.xml")
#     files = open(caltech_dir+'.xml',"r")
#     doc = ET.parse(files)
#     root = doc.getroot()
#     size_tag = root.iter("object")
#     print(size_tag)
#     for obj in size_tag:
#         print(obj.find("bndbox").findtext("xmin"))
#         print(obj.find("bndbox").findtext("xmax"))
#         print(obj.find("bndbox").findtext("ymin"))
#         print(obj.find("bndbox").findtext("ymax"))


    # for len in size:
    #     if len.find("xmin").text == "xmin":
    #         print(len.find("xmin").text)
    #         print('xmin')
    #     if len.find("xmax").text == "xmax":
    #         print(len.find("xmax").text)
    #         print('xmax')
    #
    #     if len.find("ymin").text == "ymin":
    #         print(len.find("ymin").text)
    #         print('ymin')
    #
    #     if len.find("ymax").text == "ymax":
    #         print(len.find("ymax").text)
    #         print('ymax')



#
# for f in enumerate(files):
#     doc = ET.parse(f)
#     root = doc.getroot()
#     size_tag = root.findall("object")
#     for obj in size_tag:
#         if obj.find("name").text == 'person':
#             label.write('0\n')
#             size =obj.find("bndbox")
#         elif obj.find("name").text == 'aeroplane':
#             label.write('1\n')
#         elif obj.find("name").text == 'tvmonitor':
#             label.write('2\n')
#         elif obj.find("name").text == 'train':
#             label.write('3\n')
#         elif obj.find("name").text == 'boat':
#             label.write('4\n')
#         elif obj.find("name").text == 'dog':
#             label.write('5\n')
#         elif obj.find("name").text == 'bird':
#             label.write('6\n')
#         elif obj.find("name").text == 'bicycle':
#             label.write('7\n')
#         elif obj.find("name").text == 'bottle':
#             label.write('8\n')
#         elif obj.find("name").text == 'sheep':
#             label.write('9\n')
#         elif obj.find("name").text == 'diningtable':
#             label.write('10\n')
#         elif obj.find("name").text == 'horse':
#             label.write('11\n')
#         elif obj.find("name").text == 'motorbike':
#             label.write('12\n')
#         elif obj.find("name").text == 'sofa':
#             label.write('13\n')
#         elif obj.find("name").text == 'cow':
#             label.write('14\n')
#         elif obj.find("name").text == 'cat':
#             label.write('15\n')
#         elif obj.find("name").text == 'bus':
#             label.write('16\n')
#         elif obj.find("name").text == 'pottedplant':
#             label.write('17\n')
#         elif obj.find("name").text == 'chair':
#             label.write('18\n')
#         elif obj.find("name").text == 'car':
#             label.write('19\n')
#
# print('끝났습니다')