def solution(A,B):
    answer = 0
    B = sorted(B)
    A = sorted(A,reverse=True)
    # 큰 수와 작은 수를 곱한 값이 곱할 수 있는 수중에서 가장 작은 수
    for i in range(len(A)):
        answer+=A[i]*B[i]

    return answer


A = [1, 4, 2]
B = [5,4, 4]
B = sorted(B)
A = sorted(A,reverse=True)
print(A)
print(B)