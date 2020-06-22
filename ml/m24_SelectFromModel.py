from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_boston()

# x = dataset.data
# y = dataset.target


x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.__class__)
print(x_train.shape)

print(y_train.__class__)
print(y_train.shape)

'''
model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 : ",score)

thresholds = np.sort(model.feature_importances_)

print(thresholds)


for thresh in thresholds : # 컬럼수만큼 돈다! 빙글빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    # print(selection_x_train.shape) # 칼럼이 하나씩 줄고 있는걸 알 수 있음 (가장 중요 x를 하나씩 지우고 있음)

    selection_model = XGBRegressor()
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))

# 그리드 서치 엮어라

# 데이콘 적용해라. 71개 칼럼 -> 성적 메일로 제출하기

# 메일 제목 : 말똥이 10등
'''
'''
for thresh in thresholds:   # 칼럼수만틈 돈다    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
# threshold=thresh 공부 // mdeian
    select_x_train = selection.transform(x_train)
# print(select_x_train.shape) # ctrl + space 누르면 자동완성 load

칼럼을 하나씩 빼줄건데 중요도가 낮은순으로 빠짐

    selection_model = XGBRegressor()
    selection_model.fit(selection_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print('R2는',score)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100.0))

# 그리드 서치까지 엮어라
'''