# def fibo(num):
#     if num<2:
#         return num
#     a, b = 0,1
#     for i in range(num-1):
#         a,b = b,a+b
#     # range가 num-1까지 되는 이유는 이미 0,1이 있으므로 n-1까지만
#     # 계산해도 원하는 값이 나옴
#     # 피보나치 수열의 경우 보통은 재귀적으로 풀지만 그렇게 풀게되면
#     # O(N**2)의 복잡도를 가지게 됨 따라서 반복적인 풀이로 O(N)으로 변환
#     # 추후 동적계획법을 활용한 풀이를 밑에 작성
#     return b%1234567
# def solution(n):
#     answer = fibo(n)

#     return answer

# print(solution(10))
# print(solution(5))

def dynamic_fibo(num):
    cache = [0 for _ in range(num+1)]
    cache[1] = 1

    for i in range(2, num+1):
        cache[i] = cache[i-1]%1234567 + cache[i-2]%1234567
# 하지만 나머지에 대하여 c = a+b => c%A = a%A + b%A 이므로
# 각각에 나머지를 취해준다
    return cache[num]

def solution(n):
    answer = dynamic_fibo(n)
    return answer
n =3
print(solution(n))
n=5
print(solution(n))

