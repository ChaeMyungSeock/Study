def lcs(a,b):
    if a%b == 0:
        return b
    else :
        return lcs(b, (a%b))
    
    # 완전히 나누어 떨어지면 b를 리턴
    # 아니면 b와 a%b로 다시 계속 반복하는데 
    # 최소 공배수가 1이 될 때 까지 반복
    
# def solution(arr):
#     answer = 1
#     for i in arr:
#         answer =(answer*i)//lcs(answer,i)
#     return answer

def solution(arr):
    arr.sort(reverse=True)
    print(arr)
    # arr[i]*arr[i+1] 곱해주고 최소 공배수로 나눔
    for i in range(len(arr)-1):
        arr[i+1] = arr[i]*arr[i+1]//lcs(arr[i], arr[i+1])
    print(arr)
    return arr[-1]

#     return answer

arr = [2,6,8,14]
print(solution(arr))

arr = [1,2,3]
print(solution(arr))

arr = [2, 3, 4]
print(solution(arr))

arr = [14, 2, 7]
print(solution(arr))
