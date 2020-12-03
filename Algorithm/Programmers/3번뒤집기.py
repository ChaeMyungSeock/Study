def solution(n):
    answer = 0
    str = three(n, 3)
    answer = ten(str, 3)


    return answer


def three(num,n):
    answer = ""
    while num: # num >=1
        number = num % n
        num = num // n
        answer += str(number)
    answer = int(answer)
    answer = str(answer)
    return answer

def ten(num, n):
    answer = 0
    l = len(num)
    for i in range(l):
        a = 1
        for j in range(i,l-1):
            a *= n
        answer += int(num[i]) *a

    return answer
n = 45
n1 = 125

a = solution(n1)
print(a)
