# pandas dataframe 생성
'''
DataFrame은 Series를 여러개 묶은 것 같은 2차원 데이터 구조
DataFrame의 값으로 딕셔너리형(리스트 포함)을 넣어도 됨(개당 리스트의 길이는 동일해야 함)
'''
import pandas as pd
# 행 추가
'''
행을 추가하는 예
df의 컬럼과 df에 추가할 Series형 데이터의 인덱스가 일치하지 않으면 df에 새로운 칼럼이 추가
되고, 값이 존재하지 않는 요소는 NaN으로 채워짐
'''

data = {"fruits" : ["apple", "orange", "banana","strawberry","kiwifruit"],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
# print(df)
series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])
df = df.append(series, ignore_index = True)

print(df)

# 열 추가
'''
기존의 DataFrame에 컬럼으로 추가할 수 있다. DataFrame 변수 df에 대해 'df["새로운 칼럼"]'으로
Series 또는 리스트를 대입해서 새 열을 추가할 수 잇다. 리스트를 대입하면 첫 행부터 순서대로 요소가
할당, Series를 대입하면 Series의 인덱스가 df의 인덱스에 대응
열을 추가해줄 땐 행의 갯수가 맞아야 오류가 안남
''' 
df["price"] = [150, 120, 100, 300, 150,100]
print(df)

# 데이터 참조
'''
DataFrame의 데이터는 행과 열을 지정해서 참조할 수 있음
loc은 이름을 참조, iloc는 번호로 참조
'''

# 이름으로 참조
'''
df.loc["인덱스 리스트", "컬럼 리스트"]
DataFrame 형 변수 df를 df.loc["인덱스 리스트", "컬럼 리스트"]로 지정하여 해당 범위 DataFrame를 얻을 수 있음
'''

df = df.loc[[1,2], ["time","year"]]

# 번호로 참조
'''
df.iloc["행 번호 리스트", "열 번호 리스트"]로 지정하여 해당 범위 dataFrame을 얻어옴
행 번호와 열번호는 0부터 시작
'''
df = df.iloc[[1,3],[0,2]]


# 행 또는 열 삭제
'''
DataFrame형 변수 df에 대해 df.drop()으로 인덱스 또는 컬럼을 지정하여 해당 행 또는 열을 삭제한
DataFrame을 생성할 수 있습니다. 인덱스 또는 컬럼을 리스트로 전달하여 한꺼번에 삭제할 수 있으며,
행과 열을 동시에 삭제할 수 있음. 열을 삭제하려면 두 번째 인수로 axis=1을 전달해야 함
'''
# drop()을 이용하여 df의 0,1행을 삭제합니다
df_1 = df.drop(range(0,2))

# drop()을 이용하여 df의 열 "year"를 삭제
df_2 = df.drop("year", axis = 1)

# 정렬
'''
DataFrame형 변수 df에 대해 df.sort_values(by="컬럼 또는 컬럼 리스트", ascending = True)를
지정하여 열의 값을 오름차순으로 정렬한 DataFrame을 생성
ascending = False는 내림차순 default값은 ascending = True
'''
df = df.sort_values(by = "year", ascending=True)

# 필터링
'''
DataFrame의 경우도 Series와 만차가지로 bool형의 시퀸스를 지정하여 True인 것만 추출하는 필터링을
수행할 수 있다. 또한 Series의 경우와 마찬가지로 DataFrame을 이용한 조건식으로 bool형의 시퀀스를
취득할 수 있다.
'''
print(df[df.index % 2 == 0])
# index가 0, 2, 4 짝수만 print

# 필터링을 사용해서 df의 "apple" 열이 5 이상, "kiwifruit" 열 5이상
df = df.loc[df["apple"]>=5]
df = df.loc[df["kiwifruit"]>=5]
# df = df.loc[df["apple"]>=5][df["kiwifruit"]>=5]




