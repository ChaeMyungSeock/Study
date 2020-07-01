import sys

num = int(sys.stdin.readline())
# input도 괜찮은데 readline이 빠르다고 함

cnt = 0
for i in range(num):
    
    word = input()
    # 문자열은 readline하니까 뭔가 len이랑 안맞아서 그냥 input으로 받아옴
    for j in range(len(word)):
        if j != len(word)-1:
            if word[j] == word[j+1]: # 같은 문자면 패스
                pass
            elif word[j] in word[j+1:]: # 같은 문자가 아니라면 그 뒤로 나랑 같은 문자가 있는지 확인
                break
            else: # 다 확인 후에 cnt +1
                cnt +=1

    
print(cnt)

