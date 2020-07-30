
# def fibo(num):
#     if( num<2):
#         return num 
#     elif(num>=2):
#         return fibo(num-1) + fibo(num-2)

import sys

a = int(sys.stdin.readline())

def solution(a):
    sum = 0
    for i in range(1,a+1):
        sum += i
        if(a<sum):
            b = i
            break
        if(a==sum):
            return print(1,'/',i, sep='')
    
    if((a-sum+b)<b-(a-sum+b)+1):
        return print((a-sum+b),'/',b-(a-sum+b)+1, sep='')
    else:
        return print(b-(a-sum+b)+1,'/',(a-sum+b), sep='')
solution(a)
