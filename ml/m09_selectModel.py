import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import sklearn
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv',header=0)

x = iris.iloc[:, 0:4] # 판다스에서 슬라이스 iloc 위치를 알고 있으면 됨  // loc 헤더와 인덱스 알고 있어야 함
y = iris.iloc[:,4]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 6667)

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter = 'classifier')
# all_estimators => sklearn의 모든 classifier가 있음

# name, algorithm (변수명)으로 allAlgorithms이 반환값으로 반환함
# 
for (name, algroithm) in allAlgorithms:
    model = algroithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 = ", accuracy_score(y_test, y_pred))

# 26개의 모델을 한번에 돌림
'''
Name: virginica, Length: 150, dtype: int64
AdaBoostClassifier 의 정답률 =  0.9666666666666667
BaggingClassifier 의 정답률 =  0.9666666666666667
BernoulliNB 의 정답률 =  0.3
CalibratedClassifierCV 의 정답률 =  0.9666666666666667
ComplementNB 의 정답률 =  0.6
DecisionTreeClassifier 의 정답률 =  0.9666666666666667
ExtraTreeClassifier 의 정답률 =  0.9666666666666667
ExtraTreesClassifier 의 정답률 =  0.9666666666666667
GaussianNB 의 정답률 =  1.0
GaussianProcessClassifier 의 정답률 =  0.9666666666666667
GradientBoostingClassifier 의 정답률 =  0.9666666666666667
KNeighborsClassifier 의 정답률 =  0.9666666666666667
LabelPropagation 의 정답률 =  0.9666666666666667
LabelSpreading 의 정답률 =  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 =  0.9666666666666667
LinearSVC 의 정답률 =  0.9666666666666667
LogisticRegression 의 정답률 =  0.9666666666666667
LogisticRegressionCV 의 정답률 =  0.9666666666666667
MLPClassifier 의 정답률 =  0.9666666666666667
MultinomialNB 의 정답률 =  0.8333333333333334
NearestCentroid 의 정답률 =  0.9
NuSVC 의 정답률 =  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 =  0.7333333333333333
Perceptron 의 정답률 =  0.6
QuadraticDiscriminantAnalysis 의 정답률 =  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 =  1.0
RandomForestClassifier 의 정답률 =  0.9666666666666667
RidgeClassifier 의 정답률 =  0.9
RidgeClassifierCV 의 정답률 =  0.9
SGDClassifier 의 정답률 =  0.6
SVC 의 정답률 =  0.9666666666666667
'''
print(sklearn.__version__)
'''
scikit-learn                       0.22.1 # 여기서는 안돌아가는 모델이 존재 (새로운 모델이 들어오고 안쓰는 모델이 빠지는 과정에서 안되는 모델 발생)

down grade ======>

scikit-learn                       0.20.1 # 모든 모델이 돌아가도록 다운그레이드
'''