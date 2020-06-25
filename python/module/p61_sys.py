import sys

print(sys.path)
# 여기서 나온 패스에다가 파일을 넣어주면 언제든지 당겨와서 사용가능함

from test_import import p62_import
p62_import.sum2()

print('=============================')
from test_import.p62_import import sum2
sum2()

