# 카운트 기반의 단어 표현 (Count based word Representation)

# 1. 다양한 단어의 표현 방법

## 1. 단어의 표현 방법

- 단어의 표현 방법은 크게 국소 표현(Local Representation) 방법과 분산 표현(Distributed Representation) 방법으로 나눈다.
- **국소 표현** 방법은 **해당 단어 그 자체**만 보고, 특정값을 맵핑하여 단어를 표현하는 방법이며, **분산 표현** 방법은 **그 단어를 표현하고자 주변을 참고하여 단어를 표현**하는 방법이다.

  ex ) puppy(강아지), cute(귀여운), lovely(사랑스러운)라는 단어가 있을 때, 각 단어에 1번, 2번, 3번 등과 같은 숫자를 맵핑(mapping)하여 부여한다면 이는 국소 표현 방법에 해당된다.

  ex ) puppy라는 단어 근처에는 주로 cute, lovely 라는 단어가 자주 등장하므로, puppy라는 단어는 cute, lovely한 느낌이다로 단어를 정의한다. 이것은 분산표현 방버이다.

  이렇듯 국소 표현 방법은 단어의 의미, 뉘앙스를 표현할 수 없지만, 분산 표현 방법은 단어의 뉘앙스를 표현할 수 있게 된다.

## 2. 단어 표현의 카테고리화

- Local Representation : One-hot Vector, N-gram, Count Based ( Bag of Words(DTM))
- Continuous Representation : Prediction Based ( Word2Vec ), Count Based ( Full Document(LSA), Windows(Glove) )

# 2. Bag of Words (BoW)

## 1. Bag of Words란?

- BoW란 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법이다.
- BoW를 만드는 과정
  - 우선, 각 단어에 고유한 정수 인덱스를 부여한다.
  - 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만든다.
- Example

  > 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.

  ```python
  from konlpy.tag import Okt
  import re
  okt=Okt()

  token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
  # 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.
  token=okt.morphs(token)
  # OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.

  word2index={}
  bow=[]
  for voca in token:
           if voca not in word2index.keys():
               word2index[voca]=len(word2index)
  # token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.
               bow.insert(len(word2index)-1,1)
  # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.
           else:
              index=word2index.get(voca)
  # 재등장하는 단어의 인덱스를 받아옵니다.
              bow[index]=bow[index]+1
  # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)
  print(word2index)

  # {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}

  print(bow)
  # [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
  ```

  문서1에 각 단어에 대해서 인덱스를 부여한 결과는 첫번째 출력 결과이다. 문서1의 BoW는 두번째 출력 결과이다.

  물가상승률의 인덱스는 4이며, 문서1에서 물가상승률은 2번 언급되었기 때문에 인덱스 4에 해당하는 값이 2임을 알 수 있다.

## 2. Bag of Words의 다른 예제들

- 앞서 언급했듯이, BoW에 있어서 중요한 것은 단어의 등장 빈도이다. 즉, 인덱스의 순서는 전혀 상관없다.
- BoW는 전혀 다른 문서로도, 문서1과 문서2를 합성하여 만들 수도 있다.
- BoW는 각 단어가 등장한 횟수를 수치화하는 텍스트 표현 방법이기 때문에, 주로 어떤 단어가 얼마나 등장했는지를 기준으로 문서가 어떤 성격의 문서인지를 판단하는 작업에 쓰인다.
- 즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 주로 쓰인다.

## 3. CountVectorizer 클래스로 BoW 만들기

- 사이킷 런에서는 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer 클래스를 지원한다.
- 주의할 것은 CountVectorizer는 단지 띄어쓰기만을 기준으로 단어를 자르는 낮은 수준의 토큰화를 진행하고 BoW를 만든다는 점이다.

## 4. 불용어를 제거한 BoW 만들기

- 불용어는 자연어 처리에서 별로 의미를 갖지 않는 단어들이다. BoW를 사용한다는 것은 그 문서에서 각 단어가 얼마나 자주 등장했는지를 보겠다는 것이다. 즉, BoW를 만들때 불용어를 제거하는 일은 자연어 처리의 정확도를 높이기 위해서 선택할 수 있는 전처리 기법이다.

### 1. 사용자가 직접 정의한 불용어 사용

```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
```

### 2. CountVectorizer에서 제공하는 자체 불용어 사용

```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
```

### 3. NLTK에서 지원하는 불용어 사용

```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
```

# 3. 문서 단어 행렬 (Document-Term Matrix, DTM)

- DTM을 적용하면 서로 다른 문서들을 비교할 수 있어진다.

## 1. 문서 단어 행렬의 표기법

- 문서 단어 행렬이란 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것을 말한다.
- 쉽게 생각하면 각 문서에 대한 BoW를 하나의 행렬로 만든 것으로 생각할 수 있는데, BoW와 다른 표현 방법이 아니라, BoW 표현을 다수의 문서에 대해서 행렬로 표현하고 부르는 용어이다.
- 각 문서에서 등장한 단어의 빈도를 행렬의 값으로 표기하면, 서로 다른 문서들을 비교할 수 있도록 수치화할 수 있다는 점에서 의의를 가진다.

## 2. 문서 단어 행렬의 한계

### 1. 희소 표현 (Sparse representation)

- 원-핫 벡터는 단어 집합의 크기가 벡터의 차원이 되고, 대부분의 값이 0이 된다는 특징이 있었다. 공간적 낭비와 계산 리소스 증가가 원-핫 벡터의 단점이었다.
- DTM도 마찬가지이다.
- 원-핫 벡터나 DTM과 같은 대두분의 값이 0인 표현을 희소 벡터 또는 희소 행렬이라고 부르는데, 희소 벡터는 많은 양의 저장 공간과 계산을 위한 리소스를 필요로 한다.
- 이러한 이유로 전처리를 통해 단어 집합의 크기를 줄이는 일은 BoW 표현을 사용하는 모델에서 중요할 수 있다.

### 2. 단순 빈도 수 기반 접근

- 각 문서에는 중요한 단어와 불필요한 단어들이 혼재되어 있다.
- 이를 위해 사용하는 것이 TF-IDF이다.

# 4. TF-IDF (Term Frequency-Inverse Document Frequency)

## 1. TF-IDF (단어 빈도-역 문서 빈도, Term Frequency-Inverse Document Frequency)

- TF-IDF는 단어의 빈도와 역 문서 빈도를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법이다.
- 사용방법은 우선 DTM을 만든 후, TF-IDF 가중치를 부여한다.
- TF-IDF는 주로 문서의 유사도를 구하는 작업, 검색 시스템에서 검색 결과의 중요도를 정하는 작업, 문서 내에서 특정 단어의 중요도를 구하는 작업 등에 쓰일 수 있다.

> 문서를 d, 단어를 t, 문서의 총 개수를 n이라고 표현한다.

### 1. tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.

### 2. df(t) : 특정 단어 t가 등장한 문서의 수

- 특정 단어가 각 문서, 또는 문서들에서 몇 번 등장했는지는 관심가지지 않으며, 오직 특정 단어 t가 등장한 문서의 수에만 관심을 가진다.

### 3. idf(d,t) : df(t)에 반비례하는 수

$$idf(d,t) = log(\frac{n}{1+df(t)})$$

- log와 분모에 1을 더해주는 식에 의아할 수 있다.
- log를 사용하지 않다면 총 문서의 n이 커질 수록, IDF의 값은 기하급수적으로 커지게된다.
- 불용어 등과 같이 자주 쓰이는 단어들은 비교적 자주 쓰이지 않는 단어들보다 최소 수십 배 자주 등장한다. 그런데 비교적 자주 쓰이지 않는 단어들 조차 희귀 단어들과 비교하면 또 최소 수백 배는 더 자주 등장하는 편이다.
- 이 때문에 log를 씌어주지 않으면, 희귀 단어들에 엄청난 가중치가 부여될 수 있다.
- 또한 log안의 식에서 분모에 1을 더해주는 이유는 첫번째 이유로는 특정 단어가 전체 문서에서 등장하지 않을 경우에 분모가 0이 되는 상황을 방지하기 위함이다
- TF-IDF는 모든 문서에서 자주등장하는 단어는 중요도가 낮다고 판단하며, 특정 문서에서만 자주 등장하는 단어는 중요도가 높다고 판단한다.
- 자연 로그는 로그의 밑을 자연 상수 e(e=2.718281...)를 사용하는 로그를 말한다. 자연 로그는 보통 In이라고 표현한다.
