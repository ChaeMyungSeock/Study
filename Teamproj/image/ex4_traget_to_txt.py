# 텍스트 파일로 target 값에 대한 파일을 생성해주자

f = open('./Hexapod_Bot/image/data/y_test_1.txt', mode = 'wt', encoding = 'utf-8')

for i in range(98):
    print(i)
    f.write('1'+'\n')

f.close()