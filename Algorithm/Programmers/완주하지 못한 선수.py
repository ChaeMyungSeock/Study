participant = ['leo', 'kiki', 'eden']
completion = ['eden', 'kiki']
import collections
def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer)[0]

print(solution(participant, completion))

'''
def solution(participant, completion):
    # 효율성 테스트가 들어가는 문제니까 최대한 for문이 빨리 끝나는 방식-> sort사용
    participant.sort()
    completion.sort()
    answer = ''
    for i,j in zip(participant,completion) : # 처음에 이중for문 만들었는데.. 알고보니 동시에 돌아야되더라 ㅎ 이런이런
        if i != j :
            answer += i
            break
    return answer
'''