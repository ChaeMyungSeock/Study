def solution(arr1, arr2):
    row = []
    for i in range(len(arr1)):
        yol=[]
        for j in range(len(arr2[0])):
            sum =0
            for k in range(len(arr2)):
                sum += arr1[i][k]*arr2[k][j]    
            yol.append(sum)
        row.append(yol) 
    return row

arr1 = [[1, 4], [3, 2], [4, 1]]	
arr2 = [[3, 3], [3, 3]]

print(solution(arr1, arr2))

arr1 = [[2, 3, 2], [4, 2, 4], [3, 1, 4]]
arr2 = [[5, 4, 3], [2, 4, 1], [3, 1, 1]]	
print(solution(arr1, arr2))
