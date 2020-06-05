import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기 
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=';', encoding='cp949')

count_data = wine.groupby('quality')['quality'].count()
# 판다스의 groupy는 안에 칼럼을 그룹별로
# quality칼럼안에 quality를 행별로 각개체가 얼마나 있는지 카운트해서 확인 가능
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
print(count_data)

count_data.plot()
plt.show()