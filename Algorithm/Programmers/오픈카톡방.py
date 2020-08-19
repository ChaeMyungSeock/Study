def solution(record):
    answer = []
    loglist = []
    logdict = dict()
    for i in record:
        data = i.split(sep=' ')
        if(data[0] == "Leave"):
            loglist.append([data[1], "님이 나갔습니다."])
        elif(data[0] == "Enter"):
            logdict[data[1]] = data[2]
            loglist.append([data[1], "님이 들어왔습니다."])
        elif(data[0] == "Change"):
            logdict[data[1]] = data[2]

    for j in loglist:
        answer.append(logdict[j[0]] + j[1])
    return answer