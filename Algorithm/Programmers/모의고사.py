def solution(answers):
    answer = []
    x1 = [1, 2, 3, 4, 5]
    x2 = [2, 1, 2, 3, 2, 4, 2, 5]
    x3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
    cnt = [0, 0, 0]
    b = len(answers)
    n = 0

    x1 = x1[:b]
    x2 = x2[:b]
    x3 = x3[:b]

    for j in range(b):
        if x1[j%5] == answers[j]:
            cnt[0] += 1
        if x2[j%8] == answers[j]:
            cnt[1] += 1
        if x3[j%10] == answers[j]:
            cnt[2] += 1

    for i in range(3):
        if cnt[i] == max(cnt):
            answer.append(i+1)

    return answer


answers	 = [1, 3, 2, 4, 5]
print(solution(answers))