import sys

n = int(sys.stdin.readline())

even = n//2
odd = n - even

print(odd)
print(even)
for i in range(n):
    print('* '*odd)
    print(' *'*even)
