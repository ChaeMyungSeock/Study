'''
numbers는 길이 1 이상 7 이하인 문자열입니다.
numbers는 0~9까지 숫자만으로 이루어져 있습니다.
013은 0, 1, 3 숫자가 적힌 종이 조각이 흩어져있다는 의미입니다.
'''
from itertools import permutations
from collections import Counter
def solution(numbers):
    def isPrime(num): # 에라토스테네의체
        num = int(num)
        if(num<2):
            return 0
        sqrn = int(num**0.5)
        for i in range(2, sqrn+1):
            result = num % i
            if result == 0:
                return 0
        return num
    a = []
    b = []
    for i in range(1, len(numbers)+1):
        a = list(map(''.join, permutations(numbers,i)))
    # for문을 돌리면서 순열을 리스트로 뽑아냄
    # 1개 2개... 최대 수까지
    # 리스트 원소가 소수인지 에라토스테네의 체로 확인
    # 확인되면 그 숫자를 인트로 반환해서 리스트로 받은 다음 counter로 
    # 중복되지 않게 받음
        for i in a:
                b.append(isPrime(i))
            

    return len(Counter(b))-1

# def solution(numbers):
#     answer = 0
#     numbers= sorted(numbers,reverse = True) # 가장 큰 수를 찾아보자
#     num = int(''.join(numbers))
#     def isPrime(num):
#         isprime = [True]*num
#         a = []
#         sqrn = int(num**0.5)
#         for i in range(2, sqrn+1):
#             if (isprime[i] == True):
#                 for j in range(i+i,num,i):
#                     isprime[j] = [False]
    

#     return answer

# def isPrime(num):
#     isprime = [True]*num
#     a = []
#     sqrn = int(num**0.5)
#     for i in range(2, sqrn+1):
#         if (isprime[i] == True):
#             for j in range(i+i,num,i):
#                 isprime[j] = [False]
    
#     return [i for i in range(2, num) if isprime[i] == True]



        
# a=[]
# numbers = "17"
# numbers= sorted(numbers,reverse = True) # 가장 큰 수를 찾아보자
# num = int(''.join(numbers))
# print(isPrime(100)-[2])


# # a = [3,3,7,7,7]
# # b = len(Counter(a))
# # print(b)