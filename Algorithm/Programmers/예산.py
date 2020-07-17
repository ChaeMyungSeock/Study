def solution(d, budget):
    answer = 0
    d.sort()
    while 1:
        for i in range(len(d)):
            budget -=d[i]
            if(budget>=0):
                answer +=1
            else:
                break
        break
        
    return answer