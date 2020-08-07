# 일괄적으로 파일네임 변경

import sys
from os import rename, listdir
import os
# 현재 위치의 파일 목록
file_path = './Hexapod_Bot/image/data/preview_test_1'
files = listdir(file_path)

# 파일명에 번호 추가하기
count = 0
for name in files:

    src = os.path.join(file_path, name)
    dst = str(count) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    count +=1