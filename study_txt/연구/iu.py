from PIL import Image
from autocrop import Cropper
cropper = Cropper()

for i in range(150):
    try:
        cropped_array = cropper.crop('F:/Study/IU/'+str(i)+'.jpg')
        # print(cropped_array)
        cropped_image = Image.fromarray(cropped_array)
        cropped_image.save('F:/Study/IU_cropped/'+str(i)+'.jpg')
    except AttributeError:
        print(i)