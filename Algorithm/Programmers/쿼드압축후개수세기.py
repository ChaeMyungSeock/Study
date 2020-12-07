def solution(arr):
    def qudro(y, x, n):
        # 한개짜리 일 때 리턴값 계산
        if n == 1:
            return [0, 1] if arr[y][x] == 1 else [1,0]

        # 왼쪽 오른쪽 왼쪽아래 오른쪽 아래 계산

        left_up = qudro(y, x, n//2)
        right_up = qudro(y, x + n//2, n//2)
        left_down = qudro(y + n//2, x, n//2)
        right_down = qudro(y + n//2, x + n//2, n//2)


        # 4개의 사분면이 모두 같을 경우 한개로 취급
        if left_up == right_up == right_down == left_down == [0,1] or left_up == right_up == right_down == left_down == [1,0]:
            return left_up

        else:
            # 사분면 네 개의 리스트 값을 idx별로 합한 결과
            return list(map(sum,zip(left_up, left_down, right_up, right_down)))
# list(map(sum,zip(left_up, left_down, right_up, right_down)))

    answer = qudro(0,0, len(arr))

    return answer

arr = [[1,1,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1]]

print(solution(arr))

a= solution(arr)

print(len(a))