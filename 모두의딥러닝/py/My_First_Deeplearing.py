# 딥러닝을 구동하는데 필요한 케라스 함수 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리 불러오기
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 수술 환자 데이터 불러오기
Data_set = np.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

# 파이썬은 숫자를 1부터 세지 않고 0부터 센다. 범위를 지정할 경우 콜론(:) 앞의 숫자는 범위의 맨처음을 뜻하고,
# 콜론(:) 뒤의 숫자는 이 숫자 바로앞이 범위의 마지막이라는 뜻이다.
# 속성 및 클래스 분리
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

print(X)
print(Y)

# 딥러닝은 여러 층이 쌓여 결과를 만들어 낸다.
# Sequential() 함수는 딥러닝의 구조를 한 층씩 쉽게 쌓아올릴 수 있게 해준다.
# Sequential() 함수를 선언하고 나서 model.add() 함수를 이용해 필요한 층을 차례로 추가하면 된다.
# model.add() 함수에는 Dense() 함수가 포함되어있다. dense는 조밀하게 모여있는 집합이란 뜻으로,
# 각 층이 제각각 어떤 특성을 가질지 옵션을 설정하는 역할을 한다.
# 모델을 설정하고 실행하는 부분, 딥러닝 구조를 결정
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# loss, optimizer, activation 같은 키워드는 딥러닝의 핵심이 담겨있다.
# activation: 다음 층으로 어떻게 값을 넘길지 결정하는 부분. 가장 많이 사용되는 함수 relu()와 sigmoid() 함수를 지정했다.
# loss: 한 번 신경망이 실행될 때 마다 오차 값을 추적하는 함수
# optimizer: 오차를 어떻게 줄여나갈지 정하는 함수
# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=50)

