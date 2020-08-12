def solution(bridge_length,weight,truck_weights):
    answer = 0
    bridge_on = [0]* bridge_length
    curr_weight = 0
    while truck_weights:
        answer+=1
        bridge_out = bridge_on.pop(0)
        curr_weight -= bridge_out
        if curr_weight + truck_weights[0] > weight:
            bridge_on.append(0)
        else:
            truck = truck_weights.pop(0)
            bridge_on.append(truck)
            curr_weight += truck
    while curr_weight>0:
        answer +=1
        bridge_out = bridge_on.pop(0)
        curr_weight -=bridge_out
    return answer