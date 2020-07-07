def solution(seoul):
    return '김서방은 ' + str(seoul.index("Kim"))+'에 있다'

def solution(seoul):
    a = len(seoul)
    for i in range(a):
        if(seoul[i] == "Kim"):
            answer = '김서방은 '+ str(i)+'에 있다'
    return answer