s = "aabbaccc"	


def solution(list_s):
    a = len(list_s)
    answers=[]
    if a == 1:
        return 1

    for i in range(1, a//2+1):
        answer =''
        cnt = 1
        for j in range(i,a,i):
            if list_s[j-i:j] == list_s[j:j+i]:
                cnt +=1
            else:
                if cnt == 1:
                    answer += list_s[j-i:j]
                else:
                    answer += str(cnt) + list_s[j-i:j]
                    cnt = 1
        if list_s[j:j+i] == i:
            if cnt == 1:
                answer += list_s[j:j+i]
            else:
                answer += str(cnt)+ list_s[j-i:j]
                cnt = 1
        else:
            answer += list_s[j:j+i]
        answers.append(len(answer))
    
    return min(answers)



print(solution(s))        






            
    
