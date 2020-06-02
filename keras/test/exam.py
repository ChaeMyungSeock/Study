import pandas as pd
import numpy as np

samsung = pd.read_csv('./data/csv/삼성전자 주가.csv', index_col=0, header=0, sep=',', encoding='cp949')
hite = pd.read_csv('./data/csv/하이트 주가.csv', index_col=0, header=0, sep=',', encoding='cp949')



samsung= samsung.dropna(how='all')
hite = hite.dropna()





for i in range(len(samsung.index)):
    samsung.iloc[i,0] = str(samsung.iloc[i,0])
    

# hite0602의 주식 str -> int
for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = str(hite.iloc[i,j])
   

# samsung = str(samsung_0602)
# hite0602 = str(hite0602)

# samsung_0602의 거래량 str -> int
for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))
    

# hite0602의 주식 str -> int
for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',''))
        
      
samsung = samsung.sort_values(['일자'], ascending = [True])
hite = hite.sort_values(['일자'], ascending = [True])
print(samsung)
print(hite)


samsung= samsung.values
hite = hite.values


np.save('./data/samsung_0602.npy',arr=samsung)
np.save('./data/hite0602.npy',arr=hite)
