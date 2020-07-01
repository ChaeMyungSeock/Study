
def solution(progresses, speeds):
    answer = []
    cnt = 0
    progresses = [(100-a)//b for a,b in zip(progresses,speeds)]
    for idx in range(len(progresses)):
        if progresses[cnt] < progresses[idx]:
            answer.append(idx-cnt)
            cnt = idx
    answer.append(len(progresses)-cnt)
    return answer