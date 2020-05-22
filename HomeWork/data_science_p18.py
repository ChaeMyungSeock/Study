# '#' 기호는 주석을 의미한다
# 파이썬에서 주석은 실행되지 않지만, 코드를 이해하는데 도움이 된다.

for i in [1,2,3,4,5]:
    print(i)                # 'for i' 단락의 첫 번째 줄
    for j in [1,2,3,4,5]:
        print(j)            # 'for j' 단락의 첫 번째 줄
        print(i+j)          # 'for j' 단락의 마지막 줄
    print(i)                # 'for i' 단락의 마지막 줄

print("done looping")