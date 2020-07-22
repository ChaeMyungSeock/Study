def solution(n):
    answer = ''
    while 1:
        n -=1
        # 1,2,3이기 때문에 나머지 활용하기 위해 1씩 빼줌
        # 이 부분 고려 안해서 계속 틀림...
        answer = '124'[n%3] + answer 
        
        # answer += '124'[n%3] 자리수 바뀌어서 틀림 주의!! 
        
        n //=3

        if n<=0:
            break
        
    return answer

n =10
print(solution(n))