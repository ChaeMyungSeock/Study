def solution(board):
    a = len(board)
    b = len(board[0])
    # 오른쪽 마지막에서 좌상법으로 진행
    # 1에서부터 시작하는 이유는 어떤 사각형이든 정사각형의 경우 1x1를 포함한 2x2를 최소로 생각하고
    # 확장해나아가는 코드
    for i in range(1,a):
        for j in range(1,b):
            if (board[i][j]==1):
                board[i][j] = 1 + min(board[i-1][j], board[i][j-1],board[i-1][j-1])
                # 2x2를 예로 들었을 때 1,1를 제외하고 (0,0) , (0,1) , (1,0) 모두가 1이여야 성립
                # 3x3를 예로 들자면 [[0, 1, 1, 1], [1, 1, 2, 2], [1, 2, 2, 3], [0, 0, 1, 0]]로 진행되는데
                # 앞서서 2x2로 만들어진 사각형들이 자기를 제외하고는 존재하고 있어야 3x3를 인정할 수 있음
                
    
    max_num = 0
    print(board)
    for line in board:
        max_num = max(max(line),max_num)

            

    return max_num**2
            

board = [[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,1,0]]


print(solution(board))

board = [[0,0,1,1],[1,1,1,1]]


print(solution(board))

