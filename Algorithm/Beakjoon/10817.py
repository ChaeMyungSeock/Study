import sys

A,B,C = map(int, sys.stdin.readline().split())
a = [A,B,C]
a.sort()
b = a[1]
print(b)
