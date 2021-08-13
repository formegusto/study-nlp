# 자연어 처리란?

- 자연어(natural language)란 우리가 일상 생활에서 사용하는 언어를 말한다.
- 자연어 처리(natural language processing)란 이러한 자연어의 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 일을 말한다.
- 자연어 처리는 음성 인식, 내용 요약, 번역, 사용자의 감성 분석, 텍스트 분류 작업, 질의 응답 시스템, 챗봇과 같은 곳에서 사용되는 분야이다.

# 1. 필요 프레임워크와 라이브러리

- pip install tensorflow

  텐서플로우는 구글이 2015년에 공개한 머신 러닝 오픈소스 라이브러리이다. 머신 러닝과 딥 러닝을 직관적이고 손쉽게 할 수 있도록 설계되어 있다.

```python
import tensorflow as tf
tf.__version__
```

- pip install keras

  딥 러닝 프레임워크인 텐서플로우에 대한 추상화 된 API를 제공한다. 케라스는 백엔드로 텐서플로우를 사용하며, 좀 더 쉽게 딥 러닝을 사용할 수 있게 해준다.

  순수 케라스를 keras라고 표기한다면, 텐서플로우에서 케라스 API를 사용하는 경우는 tf.keras라고 표기한다.

```python
import keras
keras.__version__
```

- pip install gensim

  젠심(Gensim)은 머신 러닝을 사용하여 토픽 모델링과 자연어 처리 등을 수행할 수 있게 해주는 오픈 소스 라이브러리 이다.

```python
import gensim
gensim.__version__
```

- pip install scikit-learn

  사이킷런(Scikit-learn)은 파이썬 머신러닝 라이브러리이다. 사이킷런을 통해 나이브 베이즈 분류, 서포트 벡터 머신 등 다양한 머신 러닝 모듈을 불러올 수 있다. 또한, 사이킷 런에는 머신러닝을 연습하기 위한 아이리스 데이터, 당뇨병 데이터 등 자체 데이터 또한 제공하고 있다.

```python
import sklearn
sklearn.__version__
```

# 2. 자연어 처리를 위한 NLTK와 KoNLPy 설치하기

- pip install nltk

  엔엘티케이는 자연어 처리를 위한 파이썬 패키지이다.

  NLTK의 기능을 제대로 사용하기 위해서는 NLTK Data라는 여러 데이터를 추가적으로 설치해야 한다.

```python
import nltk
nltk.__version__
nltk.download()
```

- pip install konlpy

  코엔엘파이는 한국어 자연어 처리를 위한 형태소 분석기 패키지이다.

```python
import konlpy
konlpy.__version__
```

# 3. 판다스(Pandas) and 넘파이(Numpy) and 맷플롭립(Matplotlib)

- 아따 데이터 분석 삼대장씨들

## 1. 판다스 (Pandas)

- 판다스(Pandas)는 파이썬 데이터 처리를 위한 라이브러리이다.
- Pandas의 경우, 주로 pd라는 명칭으로 임포트하는 것이 관례이다.
- Pandas는 총 세 가지의 데이터 구조를 사용한다.
  1. 시리즈(Series)
  2. 데이터프레임(DataFrame)
  3. 패널(Panel)

### 1. 시리즈 (Series)

- 시리즈 클래스는 1차원 배열의 값(values)에 각 값에 대응되는 인덱스(index)를 부여할 수 있는 구조를 갖고 있다.

```python
sr = pd.Series([17000, 18000, 1000, 5000],
               index=["피자", "치킨", "콜라", "맥주"])
print(sr)
```

### 2. 데이터프레임 (DataFrame)

- 데이터프레임은 2차원 리스트를 매개변수로 전달한다. 2차원이므로 행방향 인덱스(index)와 열방향 인덱스(column)가 존재한다. 즉, 행과 열을 가지는 자료구조 이다.

```python
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)
print(df)
```

> 데이터프레임의 생성

- 데이터 프레임은 리스트(List), 시리즈(Series), 딕셔너리(dict), Numpy의 ndarrays, 또 다른 데이터프레임으로 생성할 수 있다.
- 리스트 생성

  ```python
  data = [
      ['1000', 'Steve', 90.72],
      ['1001', 'James', 78.09],
      ['1002', 'Doyeon', 98.43],
      ['1003', 'Jane', 64.19],
      ['1004', 'Pilwoong', 81.30],
      ['1005', 'Tony', 99.14],
  ]
  df = pd.DataFrame(data, columns=['학번', '성명', '점수'])
  print(df)
  ```

- 딕셔너리 생성

  ```python
  # 딕셔너리로 생성하기
  data = {'학번': ['1000', '1001', '1002', '1003', '1004', '1005'],
          '이름': ['Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
          '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]}

  df = pd.DataFrame(data)
  print(df)
  ```

> 데이터프레임 조회하기

- df.head(n), df.tail(n), df['열이름']

## 2. 넘파이(Numpy)

- 넘파이(Numpy)는 수치 데이터를 다루는 파이썬 패키지이다. Numpy의 핵심이라고 불리는 다차원 행렬 자료구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에서 주로 사용된다.
- Numpy는 편의성뿐만 아니라, 속도면에서 순수 파이썬에 비해 압도적으로 빠르다는 장점이 있다.
- Numpy 주요모듈

  np.array() # 리스트 튜플 배열로 부터 ndarray를 생성

  np.asarray() # 기존 array로 부터 ndarray를 생성

  np.arange() # range와 비슷

  np.linspace(start, end, num) # [start, end] 균일한 가격으로 num개 생성

  np.logspace(start, end, num) # [start, end] log scale 간격으로 num개 생성

### 1. np.array

- np.array는 리스트, 튜플, 배열로부터 ndarray를 생성한다.

```python
a = np.array([1, 2, 3, 4, 5])  # 리스트를 가지고 1차원 배열 생성
print(type(a))
print(a)

b = np.array([[10, 20, 30], [60, 70, 80]])
print(b)  # 출력
```

- 추가적으로 행렬의 차원 및 크기를 ndim 속성과 shape 속성으로 출력할 수 있다.

```python
print("Dimensionality:", b.ndim) # 2
print("Size:", b.shape) # (2, 3)
```

### 2. ndarray의 초기화

- zeros()는 해당 배열에 모두 0을 삽입하고, ones()는 모두 1을 삽입한다. full()은 배열에 사용자가 지정한 값을 넣는데 사용한다.
- eye()는 대각선으로는 1이고 나머지는 0인 2차원 배열을 생성한다.

```python
# initialize ndarray
zero = np.zeros((3, 3))
print(zero)

ones = np.ones((4, 4))
print(ones)

full = np.full((5, 5), 7)
print(full)

eye = np.eye(6)
print(eye)

ran = np.random.random((2, 2)) # 임의의 값으로 채움
print(ran)
```

### 3. np.arange()

- 지정해준 범위에 대해서 배열을 생성한다.

```python
arange_1 = np.arange(0, 10)
print(arange_1)  # [0 1 2 3 4 5 6 7 8 9]

arange_2 = np.arange(0, 10, 2)
print(arange_2)  # [0 2 4 6 8]
```

### 4. reshape()

```python
a = np.array(np.arange(30)).reshape((5, 6))
print(a)
'''
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]
'''
```

### 5. Numpy 슬라이싱

- ndarray를 통해 만든 다차원 배열은 파이썬의 리스트처럼 슬라이스(Slice) 기능을 지원한다.

```python
b = a[0:2, 0:2]
print(b)
'''
[[0 1]
 [6 7]]
'''
```

### 6. Numpy 정수 인덱싱 (integer indexing)

```python
a = np.array([[1,2], [4,5], [7,8]])
b = a[[2, 1],[1, 0]] # a[[row2, row1],[col1, col0]]을 의미함.
print(b)
# [8 4]
```

### 7. Numpy 연산

- +, -, \*, / 의 연산자를 사용할 수 있으며, 또는 add(), subtrac, multiply(), divide() 함수를 사용할 수도 있다.

```python
x = np.array([1,2,3])
y = np.array([4,5,6])

b = x + y
# b = np.add(x, y)와 동일함
print(b)
# [5 7 9]
```

```python
b = x - y # 각 요소에 대해서 빼기
# b = np.subtract(x, y)와 동일함
print(b)
# [-3 -3 -3]
```

```python
b = b * x # 각 요소에 대해서 곱셈
# b = np.multiply(b, x)와 동일함
print(b)
# [-3 -6 -9]
```

```python
b = b / x # 각 요소에 대해서 나눗셈
# b = np.divide(b, x)와 동일함
print(b)
# [-3 -3 -3]
```

- 위에서 수행한 것은 요소별 곱으로서, Numpy에서 벡터와 행렬의 곱 또는 행렬의 고을 위해서는 dot()을 사용해야 한다.

```python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

c = np.dot(a, b)
print(c)
'''
[[19 22]
 [43 50]]
'''
```

## 3. 맷플롯립(Matplotlib)

- 맷플롯립(Matplotlib)은 데이터를 차트(chart)나 플롯(plot)으로 시각화(visualization)하는 패키지이다.
- 이는 데이터 분석 이전에 데이터 이해를 위한 시각화나, 데이터 분석 후에 결과를 시각화하기 위해서 사용된다.

### 1. 라인 플롯 그리기

- plot 메서드는 X축과 Y축의 값을 기재하고, 그림을 표시하는 show()를 통해서 시각화한다.

```python
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
plt.show()
```

### 2. 축 레이블 삽입하기

```python
plt.xlabel('hours')
plt.ylabel('score')
```

### 3. 라인 추가와 범례 삽입하기

- 하나의 plot()뿐만 아니라 여러개의 plot()을 사용하여 하나의 그래프에 나타낼 수 있다. 여러개의 라인 플롯을 동시에 사용할 경우에는 각 선이 어떤 데이터를 나타내는지를 보여주기 위해 범례(legnd)를 사용할 수 있다.

```python
plt.title('students')
plt.plot([1,2,3,4],[2,4,8,6])
plt.plot([1.5,2.5,3.5,4.5],[3,5,8,10]) #라인 새로 추가
plt.xlabel('hours')
plt.ylabel('score')
plt.legend(['A student', 'B student']) #범례 삽입
plt.show()
```

# 4. 판다스 프로파일링(Pandas-Profiling)

- 좋은 머신 러닝 결과를 얻기 위해서는 데이터의 성격을 파악하는 과정이 선행되어야 한다. 이 과정에서 데이터 내 값의 분포, 변수 간의 관계, Null 값과 같은 결측값(missing values) 존재 유무 등을 파악하게 되는데, 이와 같이 데이터를 파악하는 과정을 EDA(Exploratory Data Analysis, 탐색적 데이터 분석)이라고 한다.
- 방대한 양의 데이터를 가진 데이터프레임을 .profile_report()라는 단 한 줄의 명령으로 탐색하는 패키지인 판다스 프로파일링(pandas-profiling)을 소개한다.

## 1. 실습 파일 불러오기

[SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

```python
import pandas as pd
import pandas_profiling
data = pd.read_csv('data/spam.csv',encoding='latin1')
data
```

- 해당 데이터에서 v1의 ham은 정상 메일을 의미하고, spam은 스팸 페일을 의미한다.
- v2열은 메일의 본문을 담고 있다.

## 2. 리포트 생성하기

```python
pr=data.profile_report()
pr.to_file('./profiling/pr_report.html')
```

- 데이터프레임에 .profile_report()를 사용하여 데이터를 프로파일링한 리포트를 생성할 수 있다.

## 3. 리포트 살펴보기

```python
pr
```

> 개요 (Overview)

- 해당 데이터프레임에 전체적인 데이터 특성을 보여준다.

> 변수 (Variables)

- 데이터에 존재하는 모든 특성 변수들에 대한 결측값, 중복을 제외한 유일한 값(unique values)의 개수 등의 통계치를 보여준다. 또한 상위 5개의 값에 대해서는 바 그래프로 시각화한 결과를 제공한다.

# 5. 머신러닝 워크플로우 (Machine Learning Workflow)

- 머신 러닝 워크플로우는 흔히 아래와 같다.
  1. 수집
  2. 점검 및 탐색
  3. 전처리 및 정제 ( 테스트 데이터 → 5 )
  4. 모델링 및 훈련
  5. 평가 ( 4,5 반복 )
  6. 배포 ( 1 반복 )

## 1. 수집 (Acquisition)

- 머신 러닝을 위해서는 기계에 학습시켜야 할 데이터가 필요하다.
- 자연어 처리의 경우, 자연어 데이터를 말뭉치 또는 코퍼스(corpus)라고 부르는데 코퍼스의 의미를 풀이하면, 조사나 연구 목적에 의해서 특정 도메인으로부터 수집된 텍스트 집합을 말한다.
- 코퍼스 즉, 텍스트 데이터의 파일 형식은 txt, csv, xml 등 다양하며, 그 출처도 음성 데이터, 웹 수집기를 통해 수집된 데이터, 영화 리뷰 등 다양하다.

## 2. 점검 및 탐색 (Inspection and exploration)

- 해당 단계에서는 데이터의 구조, 노이즈 데이터, 머신 러닝 적용을 위해서 데이터를 어떻게 정제해야하는지 등을 파악해야 한다.
- 이 단계를 탐색적 데이터 분석(Exploratory Data Analysis, EDA) 단계 라고도 하는데 이는 독립 변수, 종속 변수, 변수 유형, 변수의 데이터 타입 등을 점검하며 데이터의 특징과 내재하는 구조적 관계를 알아내는 과정을 의미한다.

## 3. 전처리 및 정제 (Preprocessing and Cleaning)

- 머신 러닝 워크플로우에서 가장 까다로운 작업 중 하나이다.
- 해당 단계에서는 자연어 처리라면 토큰화, 정제, 정규화, 불용어 제거 등의 단계를 포함한다.
- 정말 까다로운 전처리의 경우에는 전처리 과정에서 머신 러닝이 사용되기도 한다.

## 4. 모델링 및 훈련 (Modeling and Training)

- 적절한 머신 러닝 알고리즘을 선택하여 모델링이 끝났다면, 전처리가 완료된 데이터를 머신 러닝 알고리즘을 통해 기계에게 학습(training)시킨다. 이를 훈련이라고도 하는데, 이 두 용어를 혼용해서 사용한다.
- 훈련이 제대로 되었다면 그 후에 우리가 원하는 태스크(task)인 기계 번역, 음성 인식, 텍스트 분류 등의 자연어 처리 작업을 수행할 수 있다.
- 여기서 주의해야할 점은 대부분의 경우에서 모든 데이터를 기계에게 학습시켜서는 안 된다는 점이다. 데이터 중 일부는 테스트용으로 남겨두고, 훈련용 데이터만 훈련에 사용해야 한다.

  → 그래야만 기계가 학습을 하고 나서, 현재 성능이 얼마나 되는지를 측정할 수 있으며, 과적합(overfitting) 상황을 막을 수 있다.

  → 데이터의 양이 충분하여 더 세부적으로 나눌 수 있다면 훈련용, 검증용, 테스트용, 데이터를 이렇게 3 가지로 나누고 훈련용 데이터만 훈련에 사용하기도 한다.

- 검증용과 테스트용의 차이는 검증용 데이터는 현재 모델의 성능. 즉, 기계가 훈련용 데이터로 얼마나 제대로 학습이 되었는지를 판단하는 용으로 사용되며, 검증용 데이터를 사용하여 모델의 성능을 개선하는데 사용된다.
- 테스트용 데이터는 모델의 최종 성능을 평가하는 데이터로 모델의 성능을 개선하는 일에 사용되는 것이 아니라, 모델의 성능을 수치화하여 평가하기 위해 사용된다.

## 5. 평가 (Evaluation)

- 기계가 다 학습이 되었다면, 테스트용 데이터로 성능을 평가하게 된다. 평가 방법은 기계가 예측한 데이터가 테스트용 데이터의 실제 정답과 얼마나 가까운지를 측정한다.

## 6. 배포 (Deployment)

- 평가 단계에서 기계가 성공적으로 훈련이 된 것으로 판단된다면, 완성된 모델이 배포되는 단계가 된다. 다만, 여기서 완성된 모델에 대한 전체적인 피드백에 대해서 모델을 변경해야하는 상황이 온다면 다시 처음부터 돌아가야 하는 상황이 올 수 있다.
