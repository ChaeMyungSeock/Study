import sys 

n= int(sys.stdin.readline())

b = list(map(int, sys.stdin.readline().split()))
max = max(b)
min = min(b)

print(min,max)