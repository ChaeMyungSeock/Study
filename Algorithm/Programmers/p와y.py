
def solution(s):
    a = len(s)
    s_list = list(s)
    cnt1 = 0
    cnt2 = 0
    for i in range(a):
        if(s_list[i] =='p' or s_list[i] == 'P'):
            cnt1 +=1
        elif(s_list[i] =='y' or s_list[i] == 'Y'):
            cnt2 +=1
    if(cnt1 == cnt2):
        answer =True
    else:
        answer = False

    return answer