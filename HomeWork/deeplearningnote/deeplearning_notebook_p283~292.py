# 통계 정보 얻기
'''
컬럼별로 데이터의 평균값, 최댓값, 최솟값 등의 통계정보를 집계할 수 있다. 
DataFrame형 변수 df를 df.describe()하여 컬럼당 데이터 수, 평균값, 최솟값,
사분위수(25%, 50%, 75%), 최댓값 정보를 포함하는 DataFrame을 반환
df_des = df.describe().loc[["mean", "max", "min"]]
'''

# DataFrame의 행간 차이와 열간 차이 구하기
'''
행간 차이를 구하는 작업은 시계열분석에서 자주 이용함. DataFrame형 변수 df에 대해
'df.diff("행 간격 또는 열 간격", axis="방향")'을 지정하면 행간 또는 열간 차이를 
계산한 DataFrame이 생성 됨. 첫 번째 인수가 양수면 이전 행과의 차이를, 음수면 
다음 행과의 차이를 구함.  axis가 0인경우 행의 방향, 1인 경우에는 열의 방향
'''

# 그룹화
'''
데이터베이스나 DataFrame의 특정 열에서 동일한 값의 행을 집계하는 것을 그룹화라고 한다.
DataFrame 변수 df에 대해 'df.groupby("컬럼")으로 지정한 컬럼을 그룹화할 수 있음
GroupBy 객체는 반환하지만 그룹화된 결과는 표시하지 않음(그룹화만 했을 뿐)
그룹화의 결과를 표시하려면 GroupBy 객체에 대해그룹의 평균을 구하는 mean(), 합을 구하는
sum()등의 통계함수 사용
'''

# 연습문제
'''
df1와 df2는 각각 야채와 과일에 대한 DataFrame. "Name", "Type", "Price"는 각각 이름,
종류(야채인지 과일인지), 가격을 나타냄 야채와 과일을 3개씩 구매 최소비용으로
'''
df1 = pd.DataFrame([["apple", "Fruit", 120],
                    ["orange", "Fruit", 60],
                    ["banana", "Fruit", 100],
                    ["pumkin", "Vegetable", 150],
                    ["potato", "Vegetable", 80]],
                    columns = ["Name","Type", "Price"])

df2 = pd.DataFrame([["onion", "Vegetable", 60],
                    ["carrot", "Vegetable", 60],
                    ["beans", "Vegetable", 100],
                    ["grape", "Fruit", 160],
                    ["kiwifruit", "Fruit", 80]],
                    columns = ["Name","Type", "Price"])

df3 = pd.concat([df1, df2], axis = 0)

# 과일을 추출하여 Price로 정렬
df_fruit = df3.loc[df3["Type"] == "Fruit"]
df_fruit = df_fruit.sort_values(by="Price")

# 야채를 추출하여 Price로 정렬
df_veg = df3.loc[df3["Type"] == "Vegetable"]
df_veg = df_fruit.sort_values(by="Price")

print(sum(df_fruit[:3]["Price"]) + sum(df_veg[:3]["Price"]))
# df_fruit[행지정][열지정].sum()으로 해도 무관
