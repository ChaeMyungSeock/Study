import pandas as pd
# DataFrame 연결

data = {"fruits" : ["apple", "orange", "banana","strawberry","kiwifruit"],
        "time" : [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)

# 인덱스나 컬럼이 일치하는 DataFrame 간의 연결
# pd.concat("DataFrame 리스트", axis=0)
'''
으로 리스트의 선두부터 순서대로 세로로 연결, axis=1로 지정하면 가로로 연결됩니다. 
세로 방향으로 연결할 때는 동일한 컬럼으로 연결되며, 가로 방향할 때는 동일한 인덱스로
연결됩니다. 그대로 연결하므로 컬럼에 중복된 값이 생길 수 있음
'''

# 연결 시 라벨 지정하기
'''
DataFrame끼리 연결하면 라벨이 중보되는 경우가 있다. 이 경우 pd.concat()에 keys를 추가하여
중복을 피할 수 있다. 그럼 중복된 라벨이 각각의 key값 밑으로 들어가서 중복된 라벨을 피할 수 있다.
df["X", "apple"]로 "X" 컬럼안의 "apple"을 참조할 수 있다
concat_df = pd.concat([df.data1,df_data2], axis=1, keys=["X","Y"])
'''

# DataFrame 결합
'''
결합, 병합이라고 불리며 결합은 Key로 불리는 열을 지정하고, 두 데이터베이스의 Key값이 일치하는 행을
옆으로 연결한다. 결합은 크게 내부 결합과 외부결합 두 가지 방법이 있다. 
'''

# 내부 결합
'''
Key 열이 공통되지 않는 행은 삭제됩니다. 또한 동일한 컬럼이지만 값이 일치하지 않는 행의 경우 
이를 남기거나 없앨 수 있다. 두 DataFrame의 "fruits"컬럼 중에서 공통되는 것만 남음 p.271
df1,df2 두 DataFrame에 대해 'pandas.merge(df1, df2, on=Key가 될 컬럼, how="inner")
key가 아니면서 이름이 같은 열은 접미사 붙음 왼쪽 df1칼럼에는 _x가 오른쪽 df2칼럼에는 _y가
붙는다
'''

# 외부 결합
'''
Key 열이 공통되지 않아도 행이 삭제되지 않고 남음. 공통되지 않은 열에는 NaN이 생성 됨
df1,df2 두 DataFrame에 대해 'pandas.merge(df1, df2, on=Key가 될 컬럼, how="outer")
key의 열의 값이 일치하지 않는 행이 삭제되지 않고 남겨져 NaN으로 채워진 열이 생성 됨 
key가 아니면서 이름이 같은 열은 접미사 붙음 왼쪽 df1칼럼에는 _x가 오른쪽 df2칼럼에는 _y가
붙는다
'''

# 이름이 다른 열을 key로 결합하기
'''
각 컬럼을 별도로 지정하면 됨
'pandas.merge(왼쪽 DF, 오른쪽 DF, left_on = "왼쪽 DF의 컬럼", right_on = "오른쪽 DF의 컬럼", how = "결합방식"'
'''

# DataFrame을 이용한 데이터 분석 - 특정 행 얻기
'''
df.head() 첫 5행을 담긴 DataFrame를 반환, 인수를 넣어주면 인수만큼 반환
df.tail() 끝 5행을 담긴 DataFrame를 반환, 인수를 넣어주면 인수만큼 반환
'''

# 계산 처리하기
'''
Pandas와 Numpy는 상호호환이 좋아서 유연한 데이터 전달이 가능하다. NumPy에서 제공하는 함수에
Series나 DataFrame을 전달하여 전체 요소를 계산할 수 있다. Numpy 배열을 받아들이는 함수에 
DataFrame을 전달하는 경우 열 단위로 정리하여 계산된다. 또한 Pandas는 Numpy처럼 브로드캐스트를
지원하므로 Pandas 간의 게산 혹은 Pandas와 정수 간의 계산을 +,-,*,/를 사용해서 유연하게 처리
브로드캐스트 - 부족한 행이나 열을 자동으로 맞춰 줌
'''

# df의 각 요소를 두 배로 만들어라
# double_df = df*2  # double_df = df + df

# df의 각 요소를 제곱하여라
# square_df = df*df   # square_df = df**2



