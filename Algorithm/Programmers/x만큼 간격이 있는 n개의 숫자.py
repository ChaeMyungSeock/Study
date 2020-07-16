def solution(x, n):
    answer = []
    for i in range(x,x+x*n,x+1):
        answer += [i]
    return answer

print(solution(10000000,1000))