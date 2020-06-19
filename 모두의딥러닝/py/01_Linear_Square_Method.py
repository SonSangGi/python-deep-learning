import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# xy의 평균 값
mx = np.mean(x)
my = np.mean(y)

# 기울기 공식의 분모
# 최소 제곱근 공식 x의 각 원소와 x의 평균값들의 차를 제곱
# sum은 Σ에 해당하는 함수
# **2는 제곱을 구하라는 의미
# for i in x는 x의 각 원소를 한 번씩 i자리에 대입하라는 의미
divisor = sum([(i - mx) ** 2 for i in x])


# 기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)

# 기울기와 y 절편 구하기
a = divisor / dividend
b = my - (mx * a)

print("기울기 a =", a)
print("y 절편 b =", b)
