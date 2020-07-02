array = [1,5,2,6,3,7,4]
commands = [[2,5,3],[4,4,1],[1,7,3]]
def solution(array, commands):
    answer = []
    for i in range(len(commands)):
        array1 =[] 
        a = commands[i][0]-1
        b = commands[i][1]-1
        c = commands[i][2]-1
        e = len(array)
        array1 = array[a:b+1]
        print('array1 :', array1)
        print('---')
        array1.sort()
        print(array1)
        print(c)
        d = array1[c]
        print('d :', d)
        answer.append(d)
    return answer
print(solution(array, commands))

