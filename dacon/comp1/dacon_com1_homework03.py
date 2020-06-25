import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, LSTM, Flatten, Conv2D,MaxPool2D
from keras.layers import Dropout
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape) # (10000,75) : x_train, test
print('test.shape :', test.shape)   # (10000,71) : x_predict
print('submmission.shape :', submission.shape)  # (10000,4)     : y_predict
# print(train.head())
# y_train = train.loc[:,'hhb':'na']

# print(y_train.head())
train = train.interpolate() # 보간법 // 선형보간  // 데이터를 많이 짜를 경우에는 선형일 확률이 높음 따라서 선형보간을 사용 => 85점 정도
test = test.interpolate()



# print(len(train.index))
# # print(train.iloc[:,1])
train = train.fillna(train.mean())
test = test.fillna(test.mean())

rho_10 = train[train['rho'].isin(['10'])]
rho_15 = train[train['rho'].isin(['15'])]
rho_20 = train[train['rho'].isin(['20'])]
rho_25 = train[train['rho'].isin(['25'])]


# rho_10_id = train_id[train_id['rho'].isin(['10'])]
# rho_15_id = train_id[train_id['rho'].isin(['15'])]
# rho_20_id = train_id[train_id['rho'].isin(['20'])]
# rho_25_id = train_id[train_id['rho'].isin(['25'])]
# # print(rho_10_id.head())
# rho_10_id = rho_10_id.iloc[:, 0]
# rho_15_id = rho_15_id.iloc[:, 0]
# rho_20_id = rho_20_id.iloc[:, 0]
# rho_25_id = rho_25_id.iloc[:, 0]
# print(rho_10_id.head())


rho_10_x = rho_10.iloc[:, :71]
rho_15_x = rho_15.iloc[:, :71]
rho_20_x = rho_20.iloc[:, :71]
rho_25_x = rho_25.iloc[:, :71]


rho_10_y = rho_10.iloc[:, -4:]
rho_15_y = rho_15.iloc[:, -4:]
rho_20_y = rho_20.iloc[:, -4:]
rho_25_y = rho_25.iloc[:, -4:]


rho_10_src = rho_10_x.iloc[:, :36]
rho_15_src = rho_15_x.iloc[:, :36]
rho_20_src = rho_20_x.iloc[:, :36]
rho_25_src = rho_25_x.iloc[:, :36]


rho_10_dst = rho_10_x.iloc[:, 36:71]
rho_15_dst = rho_15_x.iloc[:, 36:71]
rho_20_dst = rho_20_x.iloc[:, 36:71]
rho_25_dst = rho_25_x.iloc[:, 36:71]
# print(rho_10_src.head())
# print(rho_10_dst.head())

# rho_10_dst_id = rho_10_dst.iloc[:,0]
# rho_15_dst_id = rho_15_dst.iloc[:,0]
# rho_20_dst_id = rho_20_dst.iloc[:,0]
# rho_25_dst_id = rho_25_dst.iloc[:,0]
# print(rho_10_dst_id.head())



rho_10_dst = rho_10_dst.values
rho_15_dst = rho_15_dst.values
rho_20_dst = rho_20_dst.values
rho_25_dst = rho_25_dst.values
x = train.iloc[:,:71]
# print(x.head())
print(x.shape)

y = train.iloc[:,-4:]

# print(test.isnull())
x = x.values
y = y.values
# 서브밋파일 만든다.
# y_pred.to_csv(경로)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,shuffle = True,random_state = 87)

print(x_train.shape)
print(y_train.shape)



# x_train = x_train.reshape(x_train.shape[0],24)
# x_test = x_test.reshape(x_test.shape[0],24)

# 2. 모델
model = GradientBoostingRegressor()

# def tree_fit(y_train, y_test):
#     model.fit(x_train, y_train)
#     score = model.score(x_test, y_test)
#     print('score: ', score)
#     y_predict = model.predict(x_pred)
#     y_pred1 = model.predict(x_test)
#     print('mae: ', mean_absolute_error(y_test, y_pred1))
#     return y_predict

def boost_fit_acc(y_train, y_test):
    y_predict = []
    for i in range(len(submission.columns)):
        print(i)
        y_train1 = y_train[:, i]  
        model.fit(x_train, y_train1)
       
        y_test1 = y_test[:, i]
        score = model.score(x_test, y_test1)
        print('score: ', score)

        y_pred = model.predict(test)
        y_pred1 = model.predict(x_test)
        print('mae: ', mean_absolute_error(y_test, y_pred1))

        y_predict.append(y_pred)     
    return np.array(y_predict)

y_predict = boost_fit_acc(y_train, y_test).reshape(-1, 4) 



from sklearn.metrics import mean_squared_error

# mse=mean_squared_error(y_test,y_predict)
# print("mse:",mse)

# mae = mean_absolute_error(y_test,y_predict)
# print('mae : ',mae)

# result=model.predict(test)
# print(y_predict.shape)
# print(y_predict)

'''
def plot_feature_importances_caner(model):
    plt.figure(figsize= (10, 40))
    n_featrues = test.shape[1]
    plt.barh(np.arange(n_featrues), model.feature_importances_, align='center')
    # 가로 방향으로 바차트를 그림
    plt.yticks(np.arange(n_featrues), test.columns)
    # 축의 틱과 축의 틱 라벵르 편집할 수 있음
    # yticks(ticks=None, labels=None,**kwargs)
    # ticks 틱이 위치하는 리스트, 각 축의 틱을 사용하지 않기위해 빈 리스트 입력도 가능
    # 설정한 위치에 표시할 라벨을 선정한다
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.ylim(-1,n_featrues)

plot_feature_importances_caner(model)
plt.show()
# print(y_predict)
# print(y_test)

# print(y_pred.shape)
a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_predict,a)
print(y_pred)
y_pred.to_csv('./data/dacon/comp1/sub_XG.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# print(y_pred)


'''