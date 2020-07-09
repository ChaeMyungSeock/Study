import sys

a = int(sys.stdin.readline())


for i in range(a):
    c = list(map(int, sys.stdin.readline().split()))
    avg = sum(c[1:])/c[0]
    print('avg :',avg)
    cnt = 0
    k = len(c[1:])
    for j in range(1,k+1,1):
        if(c[j]>avg):
            cnt +=1
    d = (cnt/k) * 100
    print(str("%.3f" % d )+ '%')


