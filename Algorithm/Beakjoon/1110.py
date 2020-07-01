import sys

n = int(sys.stdin.readline())
a = n
cnt = 0
while 1:
    cnt += 1
    b = n//10
    c = n%10
    n = (c*10) + ((b+c)%10)

    if(n==a):
        print(cnt)
        break
