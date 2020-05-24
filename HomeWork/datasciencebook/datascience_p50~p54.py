# 3장 데이터 시각화
# matplotlib.pyplot

from matplotlib import pyplot as plt
from matplotlib import lines

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# x축에 연도, y축에 GDP가 있는 선 그래프를 만들자.
plt.plot(years, gdp,color='green', marker='o', linestyle = 'solid')

# 제목을 더하자
plt.title("Nominal GDP")

# y축에 레이블을 추가하자.
plt.ylabel("Billions of $")
plt.show()

# 막대 그래프

'''
막대 그래프(bar chart)는 이산적인(discrete) 항목들에 대한 변화르 보여 줄 때 사용하면 좋다.
'''

movies = ["Annie Hallk", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의 x 좌표는 [0, 1, 2, 3, 4] y 좌표는 [num_oscars]로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")     # 제목을 추가
plt.ylabel("# of Academy Awards")   # y축에 레이블을 추가하자

# x축 각 막대의 중앙에 영화 제목을 레이블로 추가하자,
plt.xticks(range(len(movies)), movies)
plt.show()

'''
막대 그래프를 이용하면 히스토그램(histogram)도 그릴 수 있다. 히스토그램이란 정해진 구간에서 
해당되는 항목의 개수를 보여줌으로써 값의 분포를 관찰할 수 있는 그래프의 형태이다.
'''

from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# 점수는 10점 단위로 그룹화한다. 100점은 90점대에 속한다.
histogram = Counter(min(grade // 10*10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],         # 각 막대를 오른쪽으로 5만큼 옮기고
        histogram.values(),                         # 각 막대의 높이를 정해주고
        10,                                         # 너비는 10으로 하자
        edgecolor = (0, 0, 0))                      # 각 막대의 테두리는 검은새긍로 설정하자.

plt.axis([-5, 105, 0, 5])                           # x축은 -5부터 105      y축은 0부터 5


plt.xticks([10 *i for i in range(11)])              # x축의 레이블은 0,10, .... , 100
plt.xlabel("Decile")                                # x축에 레이블을 추가하자
plt.ylabel("# of Students")                         # y축에 레이블을 추가하자
plt.title("Distribution of Exam 1 Grades")
plt.show()

'''
plt.bar의 세 번째 인자(argument)는 막대의 너비를 정한다. 여기서는 각 구간의 너비가 10이므로
막대의 너비 또한 10으로 설정(width = 10으로 해도 무관) 또, 막대들을 오른쪽으로 5씩 이동해서
(예를 들어)'10'에 해당하는 막대의 중점이 15가 되게 했다. 막대 간 구분이 되도록 각 막대의 
테두리를 검은색으로 설정, plt.axis x축 범위를 -5에서 105 (막대가 잘리지 않도록 그리기 위해서)
y축의 범위 0-5로 정했다. plt.xticks =>  x축의 레이블은 0,10, .... , 100
plt.axis를 사용할 때는 특히 신중해야함 y축을 0부터 시작하지 않으면 데이터를 정확하게 확인하기 힘듬
'''

# 선 그래프

'''
plt.plot()을 이용하면 선 그래프(line chart)를 그릴 수 있다. 
'''

variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# 한 차트에 여러 개의 선을 그리기위해 plt.plot을 여러 번 호출할 수 있다.
plt.plot(xs, variance,          'g-',    label = 'variance')            # 실선
plt.plot(xs, bias_squared,      'r-.',   label = 'bias^2')              # 일점쇄선
plt.plot(xs, total_error,       'b:',    label = 'total error')         # 점선

# 각 선에 레이블을 미리 달아 놨기 때문에
# 범례(legend)를 쉽게 그릴 수 있다.

plt.legend(loc=9)
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
plt.show()

