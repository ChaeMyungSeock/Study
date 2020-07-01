import sys

# sys.stdin.readline() 입출력에서는 무조건 사용해라 시간이 더 빠름
n = int(sys.stdin.readline())


# n부터 시작해서 1까지 -1씩 감소하면서 출력
for i in range(n,0,-1):
    print(i)