def solution(n):
    answer = 0
    for j in range(1,n+1):
        sum = 0
        # sum은 0으로 계속 초기화
        for i in range(j,n+1):
            sum += i
            if(sum>n):
                b = i
                break
            elif(sum==n):
                answer+=1
    return answer
n=15
print(solution(n))    
        
