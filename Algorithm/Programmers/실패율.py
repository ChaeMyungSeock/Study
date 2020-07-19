
N = 5
stages = [2, 1, 2, 6, 2, 4, 3, 3]

print(stages.count(2))


def solution(N, stages):
    failure = {}
    # 실패율
    answer = [0]*N
    num = len(stages)
    for i in range(1, N+1):
        if num != 0:
            cnt = stages.count(i)
            # 해당 스테이지를 플레이하고 있는 사람
            failure[i] = cnt/num
            num -= cnt
        else:
            failure[i] = 0
        failure[i]
        
    print(failure)


            # 다음 스테이지 넘어갈 때 해당 스테이지 사람 제외
    
    # 정렬
    # failure = sorted(failure, key=lambda x:failure[x], reverse=True)
    # lambdax : failure[x]이기 때문에 튜플에서 failure[x]에 해당하는 index를 리턴하게 된다
    # 그냥 failure = sorted(failure.items(), key=lambda x:x[1], reverse=True)로 해도 value값으로 정렬이 되지만 튜플로
    # 둘다 리턴되기 때문에 값을 따로 뽑아줘야함 따라서 
    failure = sorted(failure.items(), key=lambda x:x[1], reverse=True)
    
    for i in range(N):
        answer[i] = failure[i][0]

    return answer



print(solution(N, stages))

# a = {1:1,2:4,3:9}
# print(a)
# print(list(a.keys()))