def solution(n, arr1, arr2):
    answer = []
    for i in range(n):
        pre = []
        a = 0
        b = n
        a = arr1[i] | arr2[i]
        for j in range(n):
            b -=1
            if a//2**b == 1:
                pre.append('#')
                a -= 2**b
            else:
                pre.append(' ')
        
        qw = "".join(pre)
        answer.append(qw)


    return answer

n = 5
arr1 = [9, 20, 28, 18, 11]
arr2 = [30, 1, 21, 17, 28]

print(solution(n, arr1, arr2))

