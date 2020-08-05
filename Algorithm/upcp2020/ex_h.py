import sys

a = int(sys.stdin.readline())
b = list(map(int, sys.stdin.readline().split()))

cnt = sum(b)
turn = cnt //3
cnt_two = 0
if cnt %3 != 0:
    print("NO")


else:
    for i in range(len(b)):
        cnt_two += b[i]//2
    
    if cnt_two < turn:
        print('NO')
    else:
        print('YES')
