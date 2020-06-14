import numpy as np
import pandas as pd
from numpy import nan as NA
from pandas import DataFrame
# 14.1.2csv 라이브러리로 csv 만들기

# import csv

# # with 문을 사용해서 파일을 처리합니다.
# with open("csv0.csv", "w") as csvfile:
#     # writer() 매서드의 인수로 csvfile과 개행 코드(\n)를 지정합니다.
#     writer = csv.writer(csvfile, lineterminator = "\n")

#     # writerow(리스트)로 행을 추가합니다.
#     writer.writerow(["city", "year", "season"])
#     writer.writerow(["Nagano", "1998", "winter"])


# attri_data_frame1에 attri_data_frame2 행을 추가해서 출력
# import pandas as pd
# from pandas import Series, DataFrame

# attri_data1 = {"ID" : ["100", "101","102","103","104","106","108",
#                         "110","111","113"],
#                "city" : ["서울", "부산","대전","광주","서울","서울","부산",
#                         "대전","광주","서울"],
#                 "birth_year" : [1990, 1989,1992,1997,1987,1991,1988,
#                        1990,1995,1981],
#                 "name" : ["영이", "순돌","짱구","태양","션","유리","현아",
#                         "태식","민수","호식"]}
# attri_data_frame1 = DataFrame(attri_data1)

# attri_data2 = {"ID": ["107","109"],
#                 "city": ["봉화","저주"],
#                 "birth_year": [1994,1988]}
# attri_data_frame2 = DataFrame(attri_data2)

# attri_data_frame1.append(attri_data_frame2).sort_values(by="ID",ascending = True).reset_index(drop = True)

# print(attri_data_frame1.head())

# 결측치 보완 (평균값 대입법)

# fillna로 NaN 부분에 열의 평균값을 대입합니다.
# sample_data_frame.fillna(sample_data_frame.mean())


# 데이터 요약 
# 14.4.1 키별 통계량 산출
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
            "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines", "Proline"]

print(df["Alcohol"].mean())
# 중복 데이터 삭제
# drop_duplicates() 메서드 사용

# 매핑
'''
매핑은 공통의 키 역할을 하는 테이터의 값을 가져오는 처리입니다
'''
# attri_data1 = {"ID" : ["100", "101","102","103","104","106","108",
#                         "110","111","113"],
#                "city" : ["서울", "부산","대전","광주","서울","서울","부산",
#                         "대전","광주","서울"],
#                 "birth_year" : [1990, 1989,1992,1997,1987,1991,1988,
#                        1990,1995,1981],
#                 "name" : ["영이", "순돌","짱구","태양","션","유리","현아",
#                         "태식","민수","호식"]}
# attri_data_frame1 = DataFrame(attri_data1)

# city_map = { "서울" : "서울",
#             "광주" : "전라도",
#             "부산" : "경상도",
#             "대전" : "충청도"

# }
# attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
# print(attri_data_frame1)



# 구간 분할

attri_data1 = {"ID" : ["100", "101","102","103","104","106","108",
                        "110","111","113"],
               "city" : ["서울", "부산","대전","광주","서울","서울","부산",
                        "대전","광주","서울"],
                "birth_year" : [1990, 1989,1992,1997,1987,1991,1988,
                       1990,1995,1981],
                "name" : ["영이", "순돌","짱구","태양","션","유리","현아",
                        "태식","민수","호식"]}
attri_data_frame1 = DataFrame(attri_data1)

# 분할 리스트를 만듭니다.
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

# 구간 분할을 실시합니다.
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)

print(birth_year_cut_data)
print(pd.value_counts(birth_year_cut_data))