

arr = [5, 9, 7, 10]	
divisor = 5

def solution(arr, divisor):
    answer = []
    a = len(arr)
    for i in range(a):
        b = arr[i]
        if(b% divisor == 0):
            answer.append(b)
    
    if not answer:
        answer.append(-1)
    answer.sort()

    return answer

print(solution(arr,divisor))