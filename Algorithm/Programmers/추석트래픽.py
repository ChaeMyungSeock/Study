def time_find(one_time):
    time = one_time.split(" ")
    end_time = time[1]
    during = time[2]
    # print(time)

    hour, minute, second = end_time.split(':')
    hour = int(hour)*1000
    minute = int(minute) * 1000
    second = int(second[:2] + second[3:])
    # print(second)
    mlisec = hour*3600 + minute*60 + second
    # print(mlisec)
    dur_sec = during[:-1].split('.')
    # print(dur_sec)
    if (len(dur_sec)>1):
    # 미리세크 단위 포함
        during_second = int(dur_sec[0])*1000 + int(dur_sec[1] + ('0'*(3-len(dur_sec[1]))))
    else:
    # 초만 포함
        during_second = int(dur_sec[0])*1000
    # print(during_second)
    return [mlisec - during_second + 1, mlisec]
    # 시작시간부터 끝나는 시간을 리턴



def check_Num(time, lst):
#여기서 임의의 시간에서 1초동안 겹치는 숫자를 카운트
    num = 0
    # 겹치는거 카운트
    start = time
    end = time+1000
    for during in lst:
        if not( during[1]<start or during[0]>=end):
        # 만약에 데이터 끝나는 시간이 시작시간 전에 끝나면 카운트 x
        # 만약에 데이터 시작하는 시간이 end보다 나중에 시작하면 카운트 x
            num +=1
    return num

def solution(lines):
    answer = 0
    lst =[]
    count=[]
    for string in lines:
        lst.append(time_find(string))
    for i in lst:
        count.append(check_Num(i[0],lst))
        # 데이터가 시작하는 시간을 start로 선정하고 그 때 전체 데이터의
        # 시작시간 끝나는 시간 사이를 모두 체크해서 몇 개가 겹치는지 확인
        count.append(check_Num(i[1],lst))
        # 데이터가 끝나는 시간을 start로 선정하고 그 때 전체 데이터의
        # 시작시간 끝나는 시간 사이를 모두 체크해서 몇 개가 겹치는지 확인

    return max(count)