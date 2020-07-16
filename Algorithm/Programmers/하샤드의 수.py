def solution(x):
    test = x
    sum = 0
    while(x != 0):
        sum += x%10
        x //=10
    if(test % sum == 0):
        answer = True
    else:
        answer = False
    return answer