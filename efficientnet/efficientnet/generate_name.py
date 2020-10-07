import os
import glob
import numpy as np

file_path = '/data/mask/generate/generate_testmask/without_mask'

file_names = os.listdir(file_path)

print(len(file_names))
i = 0
for name in file_names:
    src = os.path.join(file_path, name)
    dst = 'without_mask' + str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1