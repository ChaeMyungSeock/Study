def solution(n):
    answer = 0
    aros = [True]*(n+1)
    aros[0] = False
    aros[1] = False
    m = int(n**0.5)
    for i in range(2,m+1,1):
        if(aros[i]==True):
            print(i)
            for j in range(i+i, n+1, i):
                aros[j] = False
    answer = sum(aros)


    return answer
'''
에라토스테네의 체 소수의 배수를 모두 배제함
그리고 소수의 제곱수만큼보다 작은 수의 소수는 배수를 고려하지 않아도 됨
'''
print(solution(120))
# for i in range(2,10,1):
#     print(i)