import numpy as np
import pandas as pd


samsung = pd.read_csv('./data/csv/samsung.csv', index_col=0, header=0, sep=',', encoding='cp949') # index_col = 0 날짜 칼럼을 index로 처리 // 첫행을 header로 처리해서 data화 x 
hite = pd.read_csv('./data/csv/hite.csv', index_col=0, header=0, sep=',', encoding='cp949')

# 두개의 앙상블을 할 때 가중치가 1:1로 나눠지기 때문에 데이터의 상관관계에 따라 결측치를 보강할지 잘라낼지 결정
# 그래서 그냥 시가로만 판단하는것도 나쁘지 않은 방법, 그리고 하이트도 6월2일자를 다 잘라서 사용 or nan => 그 전날 값으로 대체

print(samsung.head())
print(hite.head())

print(samsung)
print(hite)


# # Nan 제거
# samsung = samsung.dropna(axis=0)       # axis = 0 행을 기준으로 => 확인해라 까먹는다 자꾸

# hite = hite.fillna(method = 'bfill')   # fillna 값을 채워줌 , 전날값으로 채워주겠다 // 전날값과 큰 값 차이가 안나서
# hite = hite.dropna(axis=0)

# None 제거2
hite = hite[0:509]
samsung = samsung[0:509]

# hite.iloc[0, 1:5] = [10,20,30,40]       # index loc = iloc 1~4까지 넣어주겠다
hite.loc["2020-06-02", "고가":"거래량"] = ['100','200','300','400']     # 거래량까지 넣어주겠다
print(hite)

# 삼성과 하이트의 정렬을 오름차수능로 변경 
samsung = samsung.sort_values(['일자'], ascending=['True'])     # ascending=['True'] True일 경우 오름차순으로 정렬하고 싶다 
hite = hite.sort_values(['일자'], ascending=['True']) 


print(hite)
print(samsung)

# 콤마제거, 문자를 정수로 형변환
for i in range(len(samsung.index)):     #'37,000' -> 37000
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',', ""))

print(samsung)
print(samsung.iloc[0,0].__class__)


for i in range(len(hite.index)):     
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',', ""))

print(hite)
print(hite.iloc[1,1].__class__)

print(samsung.shape)
print(hite.shape)

samsung = samsung.values
hite = hite.values

print(hite.__class__)

np.save('./data/samsung0603.npy', arr=samsung)
np.save('./data/hite0603.npy', arr=hite)
