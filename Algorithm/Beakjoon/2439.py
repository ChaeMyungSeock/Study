import sys

num = int(sys.stdin.readline())

for i in range(num):
    print(' '*(num-(i+1))+'*'*(i+1))