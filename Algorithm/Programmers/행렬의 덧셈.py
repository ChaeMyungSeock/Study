arr1 = [[1,2],[2,3]]
arr2 = [[3,4],[5,6]]	
print(len(arr1))


def solution(arr1, arr2):
    answer = []
    tep = []
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            tep.append(arr1[i][j] + arr2[i][j])
        answer.append(tep)
        tep = []

    return answer

print(solution(arr1, arr2))

