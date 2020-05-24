# 산점도
'''
산점도(scatterplot)는 두 변수 간의 연관 관계를 보여 주고 싶을 때 적합한 그래프다. 예를 들어 각 사용자의
친구 수와 그들이 매일 사이트에서 채류하는 시간 사이의 연관성을 보여준다.
'''
from matplotlib import pyplot as plt
from matplotlib import lines

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# 각 포인트에 레이블을 달자.
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label, 
    xy=(friend_count, minute_count),        # 레이블을 데이터 포인트 근처에 두되
    xytext=(5, -5),                         # 약간 떨어뜨려 놓자
    textcoords='offset points'
    )

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()

'''
변수들끼리 비교할 때 matplotlib이 자동으로 범위를 설정하게 하면 공정한 비교를 하지 못하게 될 수 있다.
'''
test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test_1_grades")
plt.xlabel("test_2_grades")
plt.axis("equal")
plt.show()


# 여기에서 plt.axis("equal")이라는 명령을 추가하면 공정한 비교를 할 수 있게 된다

'''
matplotlib 갤러리를 살펴보면 matplotlib로 구현할 수 있는 시각의 종류에 대해 감이 잡힐것
(https://matplotlib.org/gallery.html)

seaborn은 matplotlib를 발전시킨 것으로, 더 아름답고 복잡한 시각화를 그릴 수 있게 해준다.
(https://seaborn.pydata.org/)

Altair는 최근에 나온 선언형 시각화(declarative visualization) 파이썬 라이브러리이다.
(https://altair-viz-github.io/)
'''
