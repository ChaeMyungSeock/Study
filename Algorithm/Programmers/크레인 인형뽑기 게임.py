board = [[0,0,0,0,0],[0,0,1,0,3],[ 0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]	

def solution(board, moves):
    answer = 0
    buket=[]
    b = len(moves)
    c = len(board[0])
    for j in range(b):
        for i in range(c):
            if(board[i][moves[0]-1] != 0):
                a = board[i][moves[0]-1]
                print(a)
                if(len(buket)>=1 and buket[-1] == a):
                    answer +=2
                    board[i][moves[0]-1] = 0
                    buket.pop()
                    moves.pop(0)
                    break
                buket.append(a)
                board[i][moves[0]-1] = 0
                moves.pop(0)
                break
    
    return answer

print(solution(board, moves))



