import collections
import sys

n = int(sys.stdin.readline())

for i in range(n):
    a = list(sys.stdin.readline())
    b = len(a)
    cnt = 1
    sum = 0
    for j in range(b):
        if a[j] == 'O' :
           sum += cnt
           cnt += 1
        else:
            cnt = 1
    
    print(sum)
    
           
