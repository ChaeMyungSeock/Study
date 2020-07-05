

s = 'qwer'
def solution(s):
    mid = list(s)
    a = len(mid)

    if a%2 ==1:
        answer = mid[a//2]
    else:
        answer = mid[(a//2)-1:(a//2)+1]
        answer = ''.join(answer)
    

    return answer

print(solution(s))