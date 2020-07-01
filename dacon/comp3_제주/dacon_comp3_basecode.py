import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('D:/Study/data/dacon/comp3/201901-202003.csv')
data = data.fillna('')

df = data.copy()
df = df[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
df = df.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)
df = df.loc[df['REG_YYMM']==202003]
df = df[['CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]

# print(df.__class__)
# print(df.head())

# dtypes = df.dtypes
# encoders = {}
# for column in df.columns:
#     if str(dtypes[column]) == 'object':
#         encoder = LabelEncoder()
#         encoder.fit(df[column])
#         encoders[column] = encoder
        
# df_num = df.copy()        
# for column in encoders.keys():
#     encoder = encoders[column]
#     df_num[column] = encoder.transform(df[column])
        
train_num = df.sample(frac=1, random_state=0)
train_features = train_num.drop(['AMT'], axis=1)
train_target = np.log1p(train_num['AMT'])

model = RandomForestRegressor(n_jobs=-1, random_state=0)
model.fit(train_features, train_target)



submission = pd.read_csv('data/submission.csv', index_col=0)
submission = submission.loc[submission['REG_YYMM']==202004]
submission = submission[['CARD_SIDO_NM', 'CARD_SIDO_NM']]
submission = submission.merge(df, left_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission = submission.fillna(0)
AMT = list(submission['AMT'])*2

submission = pd.read_csv('D:/Study/data/dacon/comp3/submission.csv', index_col=0)
submission['AMT'] = AMT
submission.to_csv('D:/Study/data/dacon/comp3/base_code_submission.csv', encoding='utf-8-sig')
submission.head()
