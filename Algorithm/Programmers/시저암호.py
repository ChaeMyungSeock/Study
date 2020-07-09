def solution(s, n):
    answer = ''
    s_list = list(s)
    for i in range(len(s_list)):
        if s_list[i] == ' ':
            answer += ' '
        if s_list[i].isupper(): # 대문자일 때 true return
            answer += chr((ord(s_list[i])- ord('A') + n)%26 + ord('A') )
        
        elif s[i].islower(): # 소문자일 때 true return
            answer += chr((ord(s_list[i])- ord('a') +n)%26 + ord('a') )
    return answer


# s = "AB"
# n =1 
# print(solution(s,n))

'''
def solution(s, n):
    answer = ''
    k = ord('z') - ord('a') + 1
    n = n%k
    s_list = list(s)
    for i in s_list:
        if i == ' ':
            answer += ' '
        else:
            c = ord(i) + n
            if ((c>90 and c< 96) or c >122):
                c -= k 
            answer += chr(c)
        

    return answer

'''