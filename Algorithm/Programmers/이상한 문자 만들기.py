def solution(s):
    answer = ''
    cnt = 0
    list_s = list(s)
    for i in range(len(list_s)):
        
        if(list_s[i] == ' '):
            cnt = 1
        if cnt%2 == 0:
            cnt +=1
            list_s[i] = list_s[i].upper()
        else:
            cnt +=1
            list_s[i] = list_s[i].lower()
        
    return ''.join(list_s)
