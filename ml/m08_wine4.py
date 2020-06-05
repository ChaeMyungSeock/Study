import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 와인 데이터 읽기 
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, sep=';', encoding='cp949')

y = wine['quality']
x = wine.drop('quality',axis=1) # wine에서 퀄리티를 드랍 버리겠다.

print(x.shape)
print(y.shape)

# y 레이블 축소

newlist = []
for i in list(y):
    if i<=4:
        newlist += [0]
    elif i<=7:
        newlist += [1]
    else:
        newlist += [2]
# 와인 등급을 축소 좋음 보통 안좋음 9가지 등급 => 3가지 등급 why? y데이터가 편향된 데이터 형식을 취하고 있으므로
y = newlist

x_trian, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

model = RandomForestClassifier()
model.fit(x_trian,y_train)

acc = model.score(x_test,y_test)

y_pred = model.predict(x_test)

print("acc_score : ", accuracy_score(y_test,y_pred))
print("acc : ", acc)


# 데이터가 한쪽으로 몰려있다면 수렴되는 값을 데이터로 받는다면 y값에 대한 데이터 분포를 신경써야함