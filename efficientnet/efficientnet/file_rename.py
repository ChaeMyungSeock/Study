import os
path = '/home/john/Study/pytorch_retinaface/facedetect/data/train/within_mask'
i = 0
for filename in os.listdir(path):
    # print(path+filename, '=>', path + str(cName) + str(i) + '.jpg')
    src = os.path.join(path, filename)
    dst = 'within_mask' + str(i) + '.jpg'
    dst = os.path.join(path,dst)
    os.rename(src, dst)
    i += 1

# changeName('/home/john/Study/pytorch_retinaface/facedetect/generate_mask/without_mask/','without_mask')
# changeName('/home/john/Study/pytorch_retinaface/facedetect/train_mask/within_mask/','within_mask')
