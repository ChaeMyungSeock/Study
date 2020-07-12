# 정수 제곱금 판별
def solution(n):
    answer = 0
    num = n**0.5
    num = num *10
    if num % 10 == 0:
        answer = (n**0.5 +1)**2
    else:
        answer = -1
    return answer