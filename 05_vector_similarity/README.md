# 벡터의 유사도(Vector Similarity)

- 문서의 유사도를 구하는 일은 자연어 처리의 주요 주제 중 하나이다. 사람들이 인식하는 문서의 유사도는 주로 문사들 간에 동일한 단어 또는 비슷한 단어가 얼마나 공통적으로 많이 사용되었는지에 의존한다.
- 기계도 마찬가지이다. 기계가 계산하는 문서의 유사도의 성능은 각 문서의 단어들을 어떤 방법으로 수치화하여 표현했는지, 문서 간의 단어들의 차이를 어떤 방법으로 계산했는지에 달려 있다.

# 1. 코사인 유사도 (Cosine Similarity)

## 1. 코사인 유사도 (Cosine Similarity)

- 코사인 유사도는 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도를 의미한다.
- 두 벡터의 방향이 완전히 동일한 경우에는 1의 값을 가지며, 90도의 각을 이루면 0, 180로 반대의 방향을 가지면 -1의 값을 갖게 된다.
- 즉, 결국 코사인 유사도는 -1 이상 1이하의 값을 가지며 값이 1에 가까울 수록 유사도가 높다고 판단할 수 있다.

$$similarity = \frac{\Sigma^n_{i=1} A_i*B_i}{\sqrt{\Sigma^n_{i=1}(A_i)^2}*\sqrt{\Sigma^n_{i=1}(B_i)^2}}$$

```python
doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])

print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
# 0.6666666666666667
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
# 0.6666666666666667
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도
# 1.0000000000000002
```

- 문서3은 문서2에서 단지 모든 단어의 빈도수가 1씩 증가했을 뿐이다.
- 코사인 유사도를 텍스트 전처리에 사용하는 이유는 문서의 길이가 다른 상황에서 비교적 공정한 비교를 할 수 있도록 도와주기 때문이다.
- 코사인 유사도는 벡터의 크기가 아니라, 벡터의 방향에 초점을 두기 때문이다. 코사인 유사도가 벡터의 유사도를 구하는 또 다른 방법인 내적과 가지는 차이점이다.

## 2. 유사도를 이용한 추천 시스템 구현하기

```python
# 로드 및 전처리
data = pd.read_csv('data/movies_metadata.csv', low_memory=False)
data.head(2)
data = data.head(20000)
# null 제거
data['overview'] = data['overview'].fillna('')

# 리뷰에 단어 빈도수 TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)
# (20000, 47487)
# 20000개의 영화들은 47487개의 단어들로 구성되어 있음

# 데이터 인덱스 사전
indices = pd.Series(data.index, index=data['title']).drop_duplicates()
print(indices.head())

def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴합니다.
    return data['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')
# 12481                            The Dark Knight
# 150                               Batman Forever
# 1328                              Batman Returns
# 15511                 Batman: Under the Red Hood
# 585                                       Batman
# 9230          Batman Beyond: Return of the Joker
# 18035                           Batman: Year One
# 19792    Batman: The Dark Knight Returns, Part 1
# 3095                Batman: Mask of the Phantasm
# 10122                              Batman Begins
```

# 2. 여러가지 유사도 기법

## 1. 유클리드 거리(Euclidean Distance)

- 유클리드 거리는 문서의 유사도를 구할 때 자카드 유사도나 코사인 유사도만큼 유용한 방법은 아니다
- 다차원 공간에서 두개의 점 p와 q의 거리를 계산하는 공식이다.

$$\sqrt{\Sigma^n_{i=1}(q_i-p_i)^2} = \sqrt{(q_1-p_1)^2 + (q_1-p_1)^2 + ... + (q_n-p_n)^2}$$

## 2. 자카드 유사도(Jaccard Similarity)

- A와 B 두 개의 집합이 있을 때, 합집합에서 교집합의 비율을 구한다면 두 집합 A와 B의 유사도를 구할 수 있다는 것이 자카드 유사도의 아이디어 이다.

$$\frac{|A\cap B|}{|A\cup B|} = \frac{|A\cap B|}{|A|+|B|-|A\cap B|}$$
