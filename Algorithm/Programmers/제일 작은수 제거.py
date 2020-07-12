def solution(arr):
    answer = []
    a = len(arr)
    if a != 1:
        arr.remove(min(arr))
        answer = arr
    else:
        answer.append(-1)
    return answer