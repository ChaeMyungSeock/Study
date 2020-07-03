arrangement	= "()(((()())(())()))(())"
def solution(arrangement):
    answer = 0
    arrangement = list(arrangement)
    pointer = []
    cnt = 0
    # 총 막대기 개수
    not_cnt_long=0
    # 그만 잘리는 막대기

    for i in range(len(arrangement)):
        a = arrangement.pop(0)
        if a=="(":
            cnt += 1
            not_cnt_long += 1
        else:
            if pointer[-1] == "(":
                cnt -= 1
                not_cnt_long -= 1
                cnt += not_cnt_long
            else:
                not_cnt_long -=1
                # 연속으로 두번 ")"사용되면 짧은 막대 순으로 막대기가 끝남
        pointer.append(a)
    answer = cnt
    return answer
print(solution(arrangement))