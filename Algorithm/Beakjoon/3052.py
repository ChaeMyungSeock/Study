import collections
import sys

a = list(int(sys.stdin.readline()) for _ in range(10))

for i in range(10):
    a[i] = a[i] % 42



b = collections.Counter(a)
print(len(b))