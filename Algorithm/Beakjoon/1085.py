import sys

a = list((sys.stdin.readline().split()))
sum = []
x = int(a[0])
y = int(a[1])
w = int(a[2])
h = int(a[3])

sum.append(x)
sum.append(y)
sum.append(w-x)
sum.append(h-y)
print(min(sum)) 