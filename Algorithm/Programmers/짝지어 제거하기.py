
def solution(s):
    s = list(reversed(s))
    stack = [s.pop()]
    while s:
        if(len(stack)>=1 and s[-1] == stack[-1]):
            # stack에 원소가 있고 빼야할 원소랑 비교해서 같다면 둘다 제거
            s.pop()
            stack.pop()
        else:
            stack.append(s.pop())
            # 안같으면 그냥 stack에 보관
    if len(stack) ==0:
        return 1
        # s에 원소가 없고  stack도 없다면 모두 제거 된 것이므로 return 1
    else:
        return 0
        # 제거가 안되었다면 return 0


s = "baabaa"

print(solution(s))


    
