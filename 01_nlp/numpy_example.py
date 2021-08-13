import numpy as np

a = np.array([1, 2, 3, 4, 5])  # 리스트를 가지고 1차원 배열 생성
print(type(a))
print(a)

b = np.array([[10, 20, 30], [60, 70, 80]])
print(b)  # 출력

print("Dimensionality:", b.ndim)  # 2
print("Size:", b.shape)  # (2, 3)

# initialize ndarray
zero = np.zeros((3, 3))
print(zero)

ones = np.ones((4, 4))
print(ones)

full = np.full((5, 5), 7)
print(full)

eye = np.eye(6)
print(eye)

ran = np.random.random((2, 2))  # 임의의 값으로 채움
print(ran)

arange_1 = np.arange(0, 10)
print(arange_1)  # [0 1 2 3 4 5 6 7 8 9]

arange_2 = np.arange(0, 10, 2)
print(arange_2)  # [0 2 4 6 8]

a = np.array(np.arange(30)).reshape((5, 6))
print(a)
'''
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]
'''

b = a[0:2, 0:2]
print(b)
'''
[[0 1]
 [6 7]]
'''

a = np.array([[1, 2], [4, 5], [7, 8]])
b = a[[2, 1], [1, 0]]  # a[[row2, row1],[col1, col0]]을 의미함.
print(b)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

b = x + y
# b = np.add(x, y)와 동일함
print(b)
# [5 7 9]

b = x - y  # 각 요소에 대해서 빼기
# b = np.subtract(x, y)와 동일함
print(b)
# [-3 -3 -3]

b = b * x  # 각 요소에 대해서 곱셈
# b = np.multiply(b, x)와 동일함
print(b)
# [-3 -6 -9]

b = b / x  # 각 요소에 대해서 나눗셈
# b = np.divide(b, x)와 동일함
print(b)
# [-3 -3 -3]

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

c = np.dot(a, b)
print(c)
'''
[[ 1 * 5 + 2 * 7, 1 * 6 + 2 * 8,      [[19 22]
   3 * 5 + 4 * 7, 3 * 6 + 4 * 8 ]]     [43 50]]
'''
