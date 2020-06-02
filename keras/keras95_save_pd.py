import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", index_col=None, header=0, sep=',') # sep => ,를 기준으로 데이터를 구분한다.

#  pd => loc // ioc

print(datasets)
print(datasets.__class__)



print(datasets.head())             # 위에서부터 5개
print(datasets.tail())             # 뒤에서부터 5개


print("========================")
print(datasets.values)              # 판다스를 넘파이 형태로 바꿔줌
print(datasets.values.__class__)

# 넘파이로 저장
datasets = datasets.values

np.save('./data/iris_datasets.npy',arr=datasets)
# np.save('./data/iris_y.npy',arr=y_data)
# np.save('')