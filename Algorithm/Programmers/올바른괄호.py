def solution(s):
    sum = 0
    for i in s:
        if(i=="("):
            sum +=1
        elif(i==")"):
            sum -=1
        if(sum<0):
            return False
    if(sum==0):
        return True
    else:
        return False



s = "()()"
print(solution(s))


s = "(())()"
print(solution(s))



s = ")()("
print(solution(s))