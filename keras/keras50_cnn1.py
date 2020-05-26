from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2) ,input_shape = (10,10,1)))   # (9, 9, 10)
#strides = (2,2)
# input_shape (x,y, 1 or 3) 1 => 흑백 3 => 컬러
# 사진 만장 x = (10000, 10, 10, 1) 가로 10 by 세로 10
# 10 => 필터 갯수 => 가중치의 수는 kernel_size의 각각의 매개변수를 곱하고 필터 갯수를 곱해주면 됨 => 2*2*10 (kernel_size[0]*kernel_size[1]*convolution필터의 수)
# (2,2) => kernel_size : 정수 혹인 단일 정수의 튜플/리스트. 1D컨볼루션 윈도우의 길이를 측성
# padding : 경계 처리 방법을 정의합니다. ‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
# ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
# 이미지 조각조각 짜르건 넘겨주고 증폭 짜르고 넘겨주고 증폭~ 4배증폭인듯
# (10, 10, 1) y618height, width,chnnel
model.add(Conv2D(7, (3,3)))     #(7,7,7)
# 2 by 2로 데이터를 자름
model.add(Conv2D(5, (2,2)))     # (6, 6, 5)
model.add(Conv2D(5, (2,2),strides=(2,2)))     # (3, 3, 5)
model.add(Conv2D(5, (2,2)))       # (2, 2, 5)
model.add(Conv2D(5, (2,2), padding='same'))     # (3, 3, 5)
# model.add(MaxPooling2D(pool_size=2))
# 특성중 제일 좋은 값만 반환
# model.add(Flatten())
model.add(Dense(1))
model.summary()

# MaxPooling은 해도 좋고 안해도 좋고 항상 Conv2D의 끝은 Flatten