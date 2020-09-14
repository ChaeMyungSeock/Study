import re
def solution(new_id):
    new_id = new_id.lower()
    new_id = re.sub('[^[a-z0-9-_.]','',new_id)
    c = 0
    while 1:
        if len(new_id) >=2 and new_id[c]=='[':
                new_id = new_id[:c] + new_id[c+1:]
                c -=1 
        elif len(new_id) == 1 and new_id[c] == '[':
            new_id = ""
        if c == len(new_id)-1:
            break
        c +=1
    print(new_id)
    b = 0
    while 1:
        if len(new_id)>=1 and b>=1 and new_id[b]=='.':
            if new_id[b-1] == '.':
                new_id = new_id[:b] + new_id[b+1:]
                b -=1 
        if b == len(new_id)-1:
            break
        b +=1

    a=0

    while 1:
        if a == 0 and new_id[a]=='.':
            if len(new_id)>=2:
                new_id = new_id[1:]
                a = -1
            else:
                new_id = ""
                break
        if new_id[0] != '.' :
            break
        a += 1

    if len(new_id)>=2 and new_id[-1] == '.':
        new_id = new_id[:-1]
    elif len(new_id) == 1 and new_id[-1] == '.':
        new_id = ""

    if len(new_id) == 0:
        new_id += "a"


    elif len(new_id) >=16:
        new_id = new_id[:15]
        if new_id[-1] == '.':
            new_id = new_id[:-1]
    if len(new_id)<=2:
        while 1:
            new_id += new_id[-1]
            if new_id[-1] == '.':
                new_id = new_id[:-1]
            if len(new_id) == 3:
                break
    


    return new_id

new_id ="=+[{]}:?,<>/-_.~!@#$%^&*()=+[{]}:?,<>/"		
print(solution(new_id))

# new_id ="z-+.^."	
# print(solution(new_id))

# new_id ="=.="	
# print(solution(new_id))

# new_id ="123_.def"	
# print(solution(new_id))

# new_id ="abcdefghijklmn.p"	
# print(solution(new_id))


