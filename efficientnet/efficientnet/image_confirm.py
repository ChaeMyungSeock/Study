from PIL import Image
import os, glob, numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import matplotlib.pyplot as plt


caltech_dir1 = '/home/john/Study/pytorch_retinaface/facedetect/generate_mask/without_mask'

files = glob.glob(caltech_dir1+"/*.*")

