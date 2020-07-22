def solution(p):
    answer = ''
# 균형잡힌 괄호 문자열의 인덱스를 리턴받으려고 함
    def balanced(p):
        num = 0
        temp = []
        for idx, value in enumerate(p):
            if value == ")":
                num -=1
            if value == "(":
                num +=1
            if num == 0:
                return idx
    # 올바른 괄호 문자열인지 확인하는 함수
    def is_right(string):
        temp = []
        for i in string:
            if i == "(":
                temp.append(i)
            else:
                if len(temp) == 0:
                    return False
                    # ")"가 "("보다 먼저 더 많이 나오면 옳은 문자열이 될 수 없음 
                temp.pop()
                # "("을 temp에 넣고 ")"일 때 temp 안의 ")"제거해줘서 옳은 문자열인지 판단
        if(len(temp) !=0 ):
            # temp를 대신해서 판단을 다 했을 때 안에 뭔가 남아 있다면 균형잡힌 괄호도 올바른 문자열도 아님
            return False
        else:
            return True
            
    # 빈문자열이나 문자열 전체가 올바른 문자열이면 p를 반환
    if p == "" or is_right(p): return p
    
    # 문자열 w를 균형잡힌 괄호 문자열로 분리한다
    u,v = p[:balanced(p)+1], p[balanced(p)+1:]

    # 문자열이 u가 올바른 문자일 경우
    if is_right(u):
    # v는 1단계부터 수행
        sol_v = solution(v)
        return u + sol_v 
    else:
        # 빈문자열에 첫 번째 문자로 '('
        t = "("
        # 문자열 v에 대해 1단계부터 재귀적으로 수행한 결과 문자열을 이어 붙임
        t += solution(v)
        # ')'을 다시 붙임
        t += ")"
        # u의 첫 번째와 마지막 문자 제거후 결과를 뒤집어서 뒤에 붙임
        u = list(u[1:-1])
        # u.reverse()
        # 순서를 뒤집는게 아니라 괄호 자체를 뒤집는거라 하나하나 바꿔줘야함...
        for i in range(len(u)):
            if u[i] == "(":
                u[i] = ")"
            else :
                u[i] = "("
        u = "".join(u)
        answer = t + u


        

    return answer


p = "()))((()"				

print(solution(p))
