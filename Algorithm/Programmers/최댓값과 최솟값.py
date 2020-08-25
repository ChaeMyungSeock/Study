def solution(s):
    answer
    s = list(map(int,s.split(" ")))
    s = sorted(s)
    answer = str(s[0]) + str(s[-1]) 

    return answer
s = "1 2 3 4"
s = list(map(int,s.split(" ")))
s = sorted(s)
answer = '{} {}'.format(s[0],s[-1])
print(answer)


s = "-1 -2 -3 -4"
s = list(map(int,s.split(" ")))
s = sorted(s)
answer = '{} {}'.format(s[0],s[-1])
print(answer)

# s = "-1 -1"
# print(solution(s))

s = "-1 -2 3 4"
s = list(map(int,s.split(" ")))
s = sorted(s)
answer = '{} {}'.format(s[0],s[-1])
print(answer)