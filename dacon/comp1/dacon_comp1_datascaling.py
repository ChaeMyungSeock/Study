import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

## 데이터 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0)


## 데이터 분해 및 columns이름 저장
train_col = train.columns[:-4]
test_col = test.columns
y_train_col = train.columns[-4:]

y_train = train.values[:,-4:]
train = train.values[:,:-4]
test = test.values

# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)


## NaN값이 있는 train,test값을 다시 데이터 프레임으로 감싸주기
train = pd.DataFrame(train, columns=train_col)
test = pd.DataFrame(test, columns=test_col)


train_src = train.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values # 선형보간법
# train_damp = 1
# train_damp = 625/train.values[:,0:1]/train.values[:,0:1]*(10**(625/train.values[:,0:1]/train.values[:,0:1] - 1))
train_damp = np.exp(np.pi*(25 - train.values[:,0:1])/4)
train_dst = train.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values / train_damp# 선형보간법

test_src = test.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values
# test_damp = 1
# test_damp = 625/test.values[:,0:1]/test.values[:,0:1]*(10**(625/test.values[:,0:1]/test.values[:,0:1] - 1))
test_damp = np.exp(np.pi*(25 - test.values[:,0:1])/4)
test_dst = test.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values / test_damp
print(train_dst)



max_test = 0
max_train = 0
train_fu_real = []
train_fu_imag = []
test_fu_real = []
test_fu_imag = []
train_ifu = []
test_ifu = []

rho_10 = 0
nrho_10 = 0
rho_15 = 0
nrho_15 = 0
rho_20 = 0
nrho_20 = 0
rho_25 = 0
nrho_25 = 0

for i in range(10000):
    tmp_x = 0
    tmp_y = 0
    for j in range(35):
        if train_src[i, j] == 0:
        #     tmp_x += 1
        #     train_src[i,j] = 0
            train_dst[i,j] = 0
        if train_src[i, j] - train_dst[i, j] < 0:
            train_src[i,j] = train_dst[i,j]
        if test_src[i, j] == 0:
        #     tmp_y += 1
        #     test_src[i,j] = 0
            test_dst[i,j] = 0
        if test_src[i, j] - test_dst[i, j] < 0:
            test_src[i,j] = test_dst[i,j]

    if tmp_x > max_train:
        max_train = tmp_x
    if tmp_y > max_test:
        max_test = tmp_y
    if train['rho'][i] == 10:
        rho_10 += train_dst[i,:].sum()
        nrho_10 += 1
    if train['rho'][i] == 15:
        rho_15 += train_dst[i,:].sum()
        nrho_15 += 1
    if train['rho'][i] == 20:
        rho_20 += train_dst[i,:].sum()
        nrho_20 += 1
    if train['rho'][i] == 25:
        rho_25 += train_dst[i,:].sum()
        nrho_25 += 1

    train_fu_real.append(np.fft.fft(train_dst[i]-train_dst[i].mean(), n=60).real)
    train_fu_imag.append(np.fft.fft(train_dst[i]-train_dst[i].mean(), n=60).imag)
    test_fu_real.append(np.fft.fft(test_dst[i]-test_dst[i].mean(), n=60).real)
    test_fu_imag.append(np.fft.fft(test_dst[i]-test_dst[i].mean(), n=60).imag)
print(max_train)
print(max_test)
print("RHO")
print(rho_10/nrho_10)
print(rho_15/nrho_15)
print(rho_20/nrho_20)
print(rho_25/nrho_25)
# print(train_fu_real)
# print(train_fu_imag)
# print(test_fu_real)
# print(test_fu_imag)

# max_test = 0
# max_train = 0
# train_fu_real = []
# train_fu_imag = []
# test_fu_real = []
# test_fu_imag = []

# for i in range(10000):
#     tmp_x = 0
#     tmp_y = 0
#     for j in range(35):
#         if train.iloc[i, j+1] == 0:
#             tmp_x += 1
#             train.iloc[i,j+1] = 0
#             train.iloc[i,j+36] = 0
#         if train.iloc[i, j+1] - train.iloc[i, j+36] < 0:
#             train.iloc[i,j+36] = train.iloc[i,j+1]
#         if test.iloc[i, j+1] == 0:
#             tmp_y += 1
#             test.iloc[i,j+1] = 0
#             test.iloc[i,j+36] = 0
#         if test.iloc[i, j+1] - test.iloc[i, j+36] < 0:
#             test.iloc[i,j+36] =test.iloc[i,j+1] 
#     if tmp_x > max_train:
#         max_train = tmp_x
#     if tmp_y > max_test:
#         max_test = tmp_y
#     train_fu_real.append(np.fft.ifft(train.iloc[i, 36:71]-train.iloc[i, 36:71].mean(), norm='ortho').real)
#     train_fu_imag.append(np.fft.ifft(train.iloc[i, 36:71]-train.iloc[i, 36:71].mean(), norm='ortho').imag)
#     test_fu_real.append(np.fft.ifft(test.iloc[i, 36:71]-test.iloc[i, 36:71].mean(), norm='ortho').real)
#     test_fu_imag.append(np.fft.ifft(test.iloc[i, 36:71]-test.iloc[i, 36:71].mean(), norm='ortho').imag)
# print(max_train)
# print(max_test)
# print(train_fu_real)
# print(train_fu_imag)
# print(test_fu_real)
# print(test_fu_imag)

# print(train.isnull().sum())
# train.to_csv('./example.csv',index=False)
# # print(train)

# train에서 
# train_src = train.filter(regex='_src$',axis=1).values#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
# train_dst = train.filter(regex='_dst$',axis=1).values * train.values[:,0:1] * train.values[:,0:1]#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
# test_src = test.filter(regex='_src$',axis=1).values#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values
# test_dst = test.filter(regex='_dst$',axis=1).values* test.values[:,0:1]* test.values[:,0:1]#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values

print(((train_src - train_dst) < 0).sum())
print(((test_src - test_dst) < 0).sum())

# train_src_rank = np.argsort(train_src)[::-1][:, :10]
# train_dst_rank = np.argsort(train_dst)[::-1][:, :10]
# test_src_rank = np.argsort(test_src)[::-1][:, :10]
# test_dst_rank = np.argsort(test_dst)[::-1][:, :10]
# print(np.array(train_src - train_dst)[:2,:])






small = 1e-20

# x_train = np.concatenate([train.values[:,0:1]**2, train_src/(train.values[:,0:1]**2), train_dst, train_src/(train.values[:,0:1]**2) - train_dst,train_src/(train.values[:,0:1]**2)/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src/(train.values[:,0:1]**2), test_dst, test_src/(train.values[:,0:1]**2) - test_dst,test_src/(train.values[:,0:1]**2)/(test_dst+small)], axis = 1)



x_train = np.concatenate([train.values[:,0:1]**2, train_dst, train_dst*train_damp, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag] , axis = 1)
x_pred = np.concatenate([test.values[:,0:1]**2,train_dst, test_dst*test_damp, test_src-test_dst, test_src/(test_dst+small),test_fu_real,test_fu_imag], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, test_dst, test_src - test_dst,test_src/(test_dst+small)], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small), np.log10((train_src + small)/(train_dst+small))], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, train_dst, test_src - test_dst,test_src/(test_dst+small), np.log10((test_src+small)/(test_dst+small))], axis = 1)
# print(pd.DataFrame(x_train).isnull().sum())
# print(pd.DataFrame(np.log10(train_src.values) - np.log10(train_dst.values)))

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

np.save('./dacon/comp1/x_train.npy', arr=x_train)
np.save('./dacon/comp1/y_train.npy', arr=y_train)
np.save('./dacon/comp1/x_pred.npy', arr=x_pred)

## 푸리에 변환 
# 섞여버린 파형을 여러개으 순수한 음파로 분해하는 방법

## 이미 이 데이터는 푸리에 변환이 되어있는 데이터이다?
## 각 파장의 세기 그래프 == 푸리에변환
## 분광분석법 
## IQR = 4분할
## 각 IQR*1.5의 지점을 벗어나는 값을 이상치라고 한다.