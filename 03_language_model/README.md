# 언어 모델 (Language Model)

- 언어 모델(Language Model)이란, 단어 시퀀스(문장)에 확률을 할당하는 모델을 말한다. 어떤 문장들이 있을 때, 기계가 이 문장은 적절해! 이 문장은 말이 안 돼! 라고 사람처럼 판단할 수 있다면 기계가 자연어 처리를 정말 잘 한다고 볼 수 있다. 이게 바로 언어 모델이 하는 일이다.
- 이번 챕터에서는 통계에 기반한 전통적인 언어 모델 (Statistical Language Model, SLM)에 대해서 배운다. 통계에 기반한 언어 모델은 우리가 실제 사용하는 자연어를 근사하기에는 많은 한계가 있었고, 요즘 들어 인공 신경망이 그러한 한계를 많이 해결해주면서 통계 기반 언어 모델은 많이 사용 용도가 줄었다.

# 1. 언어 모델 (Language Model)이란?

- 언어모델을 만드는 방법은 크게는 통계를 이용한 방법과 인공 신경망을 이용한 방법으로 구분할 수 있다.

## 1. 언어 모델 (Language Model)

- 언어 모델은 단어 시퀀스에 확률을 할당(assign)하는 일을 하는 모델이다. 이를 조금 풀어서 쓰면, 언어 모델은 가장 자연스러운 단어 시퀀스를 찾아내는 모델이다.
- 단어 시퀀스에 확률을 할당하게 하기 위해서 가장 보편적으로 사용되는 방법은 언어 모델이 이전 단어들이 주어졌을 때 다음 단어를 예측하도록 하는 것 이다.
- 언어 모델에 -ing를 붙인 언어 모델링 (Language Modeling)은 주어진 단어들로부터 아직 모르는 단어를 예측하는 작업을 말한다.
- 자연어 처리로 유명한 스탠포드 대학교에서는 언어 모델을 문법(grammer)이라고 비유하기도 한다. 언어 모델이 단어들의 조합이 얼마나 적절한지, 또는 해당 문장이 얼마나 적합한지를 알려주는 일을 하는 것이 마치 문법이 하는 일 같기 때문이다.

## 2. 단어 시퀀스의 확률 할당

- P는 확률을 의미한다.

1. 기계번역 (Machine Translation)

   ```python
   P(나는 버스를 탔다) > P(나는 버스를 태운다)
   ```

2. 오타 교정 (Spell Correction)

   ```python
   P(달려갔다) > P(잘려갔다)
   ```

3. 음성 인식 (Speech Recognition)

   ```python
   P(나는 메롱을 먹는다) < P(나는 메론을 먹는다)
   ```

- 언어 모델은 위와 같이 확률을 통해 보다 적절한 문장을 판단한다.

## 3. 주어진 이전 단어들로부터 다음 단어 예측하기

### A. 단어 시퀀스의 확률

- 하나의 단어를 w, 단어 시퀀스를 대문자 W라고 한다면, n개의 단어가 등장하는 단어 시퀀스 W의 확률은 아래와 같다.

$$P(W) = P(w_1,w_2,w_3,...,w_n)$$

### B. 다음 단어 등장 확률

- n-1개의 단어가 나열된 상태에서 n번째 단어의 확률은 다음과 같다.

$$P(w_n|w_1,...,w_{n-1})$$

## 4. 언어 모델의 간단한 직관

- 앞에 어떤 단어들이 나왔는지 고려하여 후보가 될 수 있는 여러 단어들에 대해서 등장 확률을 추정하고, 가장 높은 확률을 가진 단어를 선택한다.

# 2. 통계적 언어 모델 (Statistical Language Model, SLM)

## 1. 조건부 확률

- 조건부 확률은 두 확률 P(A), P(B)에 대해서 아래와 같은 관계를 갖는다.

  $$p(B|A)=P(A,B)/P(A)$$

  $$P(A,B) = P(A)P(B|A)$$

- 더 많은 확률에 대해서 일반화를 하게되면,

  $$P(A,B,C,D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C)$$

  이를 조건부 확률의 연쇄 법칙(chain rule)이라고 한다.

## 2. 문장에 대한 확률

> P(An adorable little boy is spreading smiles)

- 각 단어는 문맥이라는 관계로 인해 이전 단어의 영향을 받아 나온 단어이다. 그리고 모든 단어로부터 하나의 문장이 완성된다.

$$P(An\ adorable\ little\ boy\ is\ spreading\ smiles) = P(An)*P(adorable|An)*P(little|An\  adorable)*P(boy|An\ adorable\ little)*...$$

## 3. 카운트 기반의 접근

- 문장의 확률을 구하기 위해서 다음 단어에 대한 예측 확률을 모두 곱한다는 것을 알았다.
- 그렇다면 SLM은 이전 단어로부터 다음 단어에 대한 확률은 어떻게 구할까? 정답은 카운트에 기반하여 확률을 계산한다.

$$P(is|An\ adorable\ little\ boy) = \frac{count(An\ adorable\ little\  boy\ is)}{count(An\ adorable\ little\ boy)}$$

- 기계가 학습한 코퍼스 데이터에서 An adorable little boy가 100번 등장했는데, 그 다음에 is가 등장한 경우는 30번일 때, 이 경우 P(is|An adorable little boy)는 30%이다.

## 4. 카운트 기반 접근의 한계 - 희소 문제 (Sparsity Problem)

- 언어 모델은 실생활에서 사용되는 언어의 확률 분포를 근사 모델링 한다.
- 기계에게 많은 코퍼스를 훈련시켜서 언어 모델을 통해 현실에서의 확률 분포를 근사하는 것이 언어 모델의 목표이다.
- 그런데 카운트 기반으로 접근하려고 한다면 갖고 있는 코퍼스(corpus). 즉, 다시 말해 기계가 훈련하는 데이터는 정말 방대한 양이 필요하다.

$$P(is|An\ adorable\ little\ boy) = \frac{count(An\ adorable\ little\  boy\ is)}{count(An\ adorable\ little\ boy)}$$

- 위와 같이 P(is|An adorable little boy)를 구하는 경우에서 기계가 훈련한 코퍼스에 An adorable little boy is 라는 단어 시퀀스가 없었다면, 이 단어 시퀀스에 대한 확률은 0이 된다. 또는 An adorable little boy라는 단어 시퀀스가 없었다면 분모가 0이 되어 확률은 정의되지 않는다.
- 이와 같이 충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못하는 문제를 희소 문제(sparsity problem)라고 한다.
- 이 문제를 완화하는 방법으로 n-gram이나, 스무딩이나 백오프 같은 여러가지 일반화(generalization) 기법이 존재한다. 하지만 희소 문제에 대한 근본적인 해결책은 되지 못하였다.
- 결국 이러한 한계로 인해 언어 모델의 트렌드는 통계적 언어 모델에서 인공 신경망 언어 모델로 넘어가게 된다.

# 3. N-gram 언어 모델 (N-gram Language Model)

- n-gram 언어 모델은 여전히 카운트에 기반한 통계적 접근을 사용하고 있으므로, SLM의 일종이다. 다만, 앞서 배운 언어 모델과는 달리 이전에 등장한 모든 단어를 고려하는 것이 아니라, 일부 단어만 고려하는 접근 방법을 사용한다. (n 이 가지는 의미)

## 1. 코퍼스에서 카운트하지 못하는 경우의 감소

- SLM의 한계는 훈련 코퍼스에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있다는 점이다.

  → 카운트할 수 없을 가능성이 높다.

- 그래서 다음과 같이 참고하는 단어들을 줄이는 것이다.

  $$P(is|An\ adorable\ little\ boy) \approx P(is|boy)$$

  → 이렇게 단어들을 줄이면, 카운트를 할 수 있을 가능성이 높아진다.

- 다른 관점으로 An adrable little boy가 나왔을 때, is가 나올 확률을 그냥 boy가 나왔을 때 is가 나올 확률로 생각해보는 것은 어떨까? 갖고 있는 코퍼스에 An adrable little boy is가 있을 가능성보다는 boy is 라는 더 짧은 단어 시퀀스가 존재할 가능성이 더 높다.
- 즉, 앞에서는 An adrable little boy가 나왔을 때, is가 나올 확률을 구하기 위해서는 An adorable little boy가 나온 횟수와 An adorable little boy is가 나온 횟수를 카운트해야만 했지만, 이제는 단어의 확률을 구하고자 기준 단어의 앞 단어를 전부 포함해서 카운트하는 것이 아니라, 앞 단어 중 임의의 개수만 포함해서 카운트하여 근사하자는 것이다.

## 2. N-gram

- 이 때, 임의의 개수를 정하기 위한 기준을 위해 사용하는 것이 n-gram이다. n-gram은 n개의 연속적인 단어 나열을 의미한다. 갖고 있는 코퍼스에서 n개의 단어 뭉치 단위로 끊어서 이를 하나의 토큰으로 간주한다.
- **uni**grams : an, adorable, little, boy, is, spreading, smiles
- **bi**grams : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
- **tri**grams : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
- **4-**grams : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles
- n-gram을 통한 언어 모델에서는 다음에 나올 단어의 예측은 오직 n-1개의 단어에만 의존한다.

  → An adorable little boy is spreading 다음에 나올 단어를 예측하고 싶다고 할 때, 4-gram을 이용한 언어 모델을 사용한다면, 이 경우 spreading 다음에 올 단어를 예측하는 것은 n-1에 해당되는 앞의 3개의 단어만을 고려한다.

  $$P(w|boy\ is\ spreading) = \frac{count(boy\ is\ spreading\ w)}{count(boy\ is\ spreading)}$$

## 3. N-gram Language Model의 한계

- n-gram은 뒤의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생긴다는 점이다. 문장을 읽다 보면 앞 부분과 뒷 부분의 문맥이 전혀 연결 안되는 경우도 생길 수 있다.
- 결론만 말하면, 전체 문장을 고려한 언어 모델보다는 정확도가 떨어질 수 밖에 없다.

> 희소 문제(Sparsity Problem)

- 문장에 존재하는 앞에 나온 단어를 모두 보는 것보다 일부 단어만을 보는 것으로 현실적으로 코퍼스에서 카운트 할 수 있는 확률을 높일 수는 있지만, 여전히 희소문제가 존재한다.

> n을 선택하는 것은 trade-off 문제

- n을 작게 선택하면 훈련 코퍼스에서 카운트는 잘 되겠지만, 근사의 정확도는 현실의 확률분포와 멀어진다. 그렇기 때문에 적절한 n을 선택해야 한다. 앞서 언급한 trade-off 문제로 인해 정확도를 높이려면 n은 최대 5를 넘게 잡아서는 안 된다고 권장되고 있다.

## 4. 인공 신경망을 이용한 언어 모델 (Neural Network Based Language Model)

# 4. 한국어에서의 언어 모델 (Language Model for Korean Sentences)

## 1. 한국어는 어순이 중요하지 않다.

## 2. 한국어는 교착어이다.

## 3. 한국어는 띄어쓰기가 제대로 지켜지지 않는다.

# 5. 펄플렉서티(Perplexity)

- 두 개의 모델 A,B 가 있을 때 이 모델의 성능은 어떻게 비교할까?
- 일일히 모델들에 대해서 실제 작업을 시켜보고 정확도를 비교하는 외부 평가(extrinsic evaluation)도 존재하지만, 조금은 부정확할 수는 있어도 테스트 데이터에 대해서 빠르게 식으로 계산되는 더 간단한 평가 방법이 존재한다.
- 바로 모델 내에서 자신의 성능을 수치화하여 결과를 내놓는 내부 평가(Intrinsic evaluation)에 해당되는 펄플렉서티(perplexity)이다.

## 1. 언어 모델의 평가 방법(Evaluation metric) : PPL

- 펄플렉서티(perplexity)는 언어 모델을 평가하기 위한 내부 평가 지표이다. 보통 줄여서 PPL이라고 표현한다.
- PPL은 수치가 높으면 좋은 성능을 의미하는 것이 아니라, 낮을수록 언어 모델의 성능이 좋다는 것을 의미한다.
- PPL은 단어의 수로 정규화(normalization)된 테스트 데이터에 대한 확률의 역수이다. PPL을 최소화한다는 것은 문장의 확률을 최대화하는 것과 같다.
- 문장 W의 길이가 N이라고 하였을 때 PPL은 다음과 같다.

  $$PPL(W) = P(w_1, w_2, ...,w_N)^{-\frac{1}{N}} = \sqrt{\frac{1}{P(w_1,w_2,...,w_N)}}^N$$

  → 해당 식에는 n-gram을 적용할 수도 있다.

## 2. 분기 계수 (Branching factor)

- PPL은 선택할 수 있는 가능한 경우의 수를 의미하는 분기계수(branching factor)이다. PPL은 이 언어 모델이 특정 시점에서 평균적으로 몇 개의 선택지를 가지고 고민하고 있는지를 의미한다.
- 가령, 언어 모델에 어떤 테스트 데이터를 주고 측정했더니 PPL이 10이 나왔다고 해보자. 그렇다면 해당 언어 모델은 테스트 데이터에 대해서 다음 단어를 예측하는 모든 시점(time-step)마다 평균적으로 10개의 단어를 가지고 어떤 것이 정답인지 고민하고 있다고 볼 수 있다.
- 단, 평가 방법에 있어서 주의할 점은 PPL의 값이 낮다는 것은 테스트 데이터 상에서 높은 정확도를 보인다는 것이지, 사람이 직접 느끼기에 좋은 언어 모델이라는 것을 반드시 의미하진 않는다는 점이다.

# 6. 조건부 확률 (Conditional Probability)

[위키독스](https://wikidocs.net/21681)

1. 학생을 뽑았을 때, 남학생일 확률

   $$P(A) = 180/360 = 0.5$$

2. 학생을 뽑았을 때, 고등학생이면서 남학생일 확률

   $$P(A\cap D) = 80/360$$

3. 고등학생 중 한명을 뽑았을 때, 남학생일 확률

   $$P(A|D) = 80/200 = P(A\cap D)/P(D)=(80/360)/(200/36)=80/200=0.4$$
