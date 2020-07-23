def solution(skill, skill_trees):
    answer = 0
    for copy_skill in skill_trees:
# skill 하나씩 불러와서 비교
        skills = list(skill)
        for i in copy_skill:
            
            if i in skill:
            # 불러온 스킬의 요소가 테크트리의 스킬일 때
                if i != skills.pop(0):
                    break
        else:
            answer +=1
                


    return answer

skill = "CBD"
skill_trees	 = ["BACDE", "CBADF", "AECB", "BDA"]










# def solution(skill, skill_trees):
#     answer = 0
#     for skills in skill_trees:
#         sk = []
#         tre = True
#         for j in range(len(skills)):
#             if skills[j] in skill:
#                 sk.append(skills[j])
# # 선행스킬에 필요한 스킬만 따로 리턴
#         for k in range(len(sk)):
#             if sk[k] != skill[k]:
#                 tre = False
#                 break
#         if tre == True:
#             answer +=1

        
#     return answer

