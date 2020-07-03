a=5
b=24


def solution(a, b):
    answer = ''
    month = [31,29,31,30,31,30,31,31,30,31,30,31]
    day = ["FRI","SAT","SUN","MON","TUE","WED","THU"]
    sum = 0
    for i in range(a-1):
        sum += month[i]

    answer = day[((sum+b)%7)-1]


    return answer


print(solution(a,b))