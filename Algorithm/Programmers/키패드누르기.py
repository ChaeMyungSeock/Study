# 왼손 : * // 오른손 : #
# 무조건 왼손 1,4,7
# 무조건 오른손 3,6,9
# 2,5,8,0 가까운 손가락이 감 하지만 같다면 편한 손가락
# 2 L => 1, 4, 7 => 1,2,3
#   R => 3, 6, 9 => 1,2,3

# 5 L => 1, 4, 7 => 2,1,2
#   R => 3, 6, 9 => 2,1,2

# 8 L => 1, 4, 7 => 3,2,1
#   R => 3, 6, 9 => 3,2,1

# 0 L => 1, 4, 7 => 4,3,2
#   R => 3, 6, 9 => 4,3,2

def get_distance(hand, number):
    location = {"1" : (0,0) , "2" : (0,1) , "3" : (0,2),
                "4" : (1,0) , "5" : (1,1) , "6" : (1,2),
                "7": (2, 0), "8": (2, 1), "9": (2, 2),
                "*": (2, 0), "0": (2, 1), "#": (2, 2)
                }

    number = str(number)
    x_hand, y_hand = location[hand]

    x_number, y_number = location[number]

    return abs(x_hand - x_number) + abs(y_hand - y_number)




def solution(numbers, hand):
    answer = ''
    left, right = "*", "#"
    hand = "R" if hand == "right" else "L"



    for i in numbers:
        if i in [1,4,7]:
            answer += "L"
            left = str(i)
        if i in [3,6,9]:
            answer += "R"
            right = str(i)
        if i in [2,5,8,0]:
            dis1 = get_distance(left, i)
            dis2 = get_distance(right, i)

            if dis1 > dis2:
                answer += "R"
                right = str(i)
            elif dis1 < dis2:
                answer += "L"
                left = str(i)
            else:
                answer += hand
                if hand == "R":
                    right = str(i)
                else:
                    left = str(i)
    return answer

numbers = [1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5]

hand = "right"

