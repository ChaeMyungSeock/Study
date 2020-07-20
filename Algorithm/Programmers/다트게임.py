'''
import re

def solution(dartResult):
    number = re.findall("\d+",dartResult)
    sam1 = re.compile("[^0-4-10]")
    sam = "".join(sam1.findall(dartResult))

    sam = list(sam)
    a = len(number)
    b = len(sam)
    sub = [0]*a
    print(number)
    print(sam)
    for i in range(a):
        sub[i] = int(number[i])
        for j in range(b-1):
            if(sam[j] == 'S'):
                sub[i] = sub[i]
                del sam[0]
                if(len(sam) == 0):
                    break
                else :
                    if(sam[j] == '*'):
                        del sam[0]
                        if(i==0):
                            sub[i] = sub[i]*2
                        else:
                            sub[i-1] = sub[i-1]*2 
                            sub[i] =  sub[i]*2
                        break
                    elif(sam[j] == '#'):
                        del sam[0]
                        sub[i] = -sub[i]
                        break
                    else : break



            elif(sam[j] == 'D'):
                sub[i] = sub[i]**2
                del sam[0]
                if(len(sam) == 0):
                    break
                else :
                    if(sam[j] == '*'):
                        del sam[0]
                        if(i==0):
                            sub[i] = sub[i]*2
                        else:
                            sub[i-1] = sub[i-1]*2 
                            sub[i] =  sub[i]*2
                        break

                    elif(sam[j] == '#'):
                        del sam[0]
                        sub[i] = -sub[i]

                        break
                    else : break





            elif(sam[j] == 'T'):
                del sam[0]
                sub[i] = sub[i]**3
                if(len(sam) == 0):
                    break
                else :
                    if(sam[j] == '*'):
                        del sam[0]
                        if(i==0):
                            sub[i] = sub[i]*2
                        else:
                            sub[i-1] = sub[i-1]*2 
                            sub[i] =  sub[i]*2
                        break


                    elif(sam[j] == '#'):
                        del sam[0]
                        sub[i] = -sub[i]
                        j +=1
                        break
                    else : break

    print(sub)
    return sum(sub)
'''
import re
def solution(dartResult):
    answer = 0
    funct = re.compile("(\d+)([A-Z])(\*|#)?")
    scores = funct.findall(dartResult)
    print(scores.__class__)
    b = len(scores)
    result = [0] * b
    for i in range(b):
        
        num = int(scores[i][0])

        if(scores[i][1] == 'S'):
            result[i] = num
            if(scores[i][2] == '*'):
                if(i == 0):
                    result[i] *=2
                else :
                    result[i-1] *=2
                    result[i] *=2
            elif(scores[i][2] == '#'):
                result[i] = -result[i]

        elif(scores[i][1] == 'D'):
            result[i]= num**2
            if(scores[i][2] == '*'):
                if(i == 0):
                    result[i] *=2
                else :
                    result[i-1] *=2
                    result[i] *=2
            elif(scores[i][2] == '#'):
                result[i] = -result[i]
        
        elif(scores[i][1] == 'T'):
            result[i] = num**3
            if(scores[i][2] == '*'):
                if(i == 0):
                    result[i] *=2
                else :
                    result[i-1] *=2
                    result[i] *=2
            elif(scores[i][2] == '#'):
                result[i] = -result[i]
            

        


    
    return sum(result)

dartResult	= "1S2D*3T"

funct = re.compile("(\d+)([A-Z])(\*|#)?")
scores = funct.findall(dartResult)
result = []
print(scores)
print(len(scores))
print(int(scores[1][0]))
print(solution(dartResult))