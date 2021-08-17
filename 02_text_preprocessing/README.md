# 텍스트 전처리 (Text preprocessing)

- 자연어 처리에 있어서 텍스트 전처리는 매우 중요한 작업이다. 텍스트 전처리는 용도에 맞게 텍스트를 사전에 처리하는 작업이다.
- 텍스트에 대해서 제대로 된 전처리를 하지 않으면, 뒤에서 배울 자연어 처리 기법들이 제대로 동작하지 않는다.

# 1. 토큰화 (Tokenization)

- 자연어 처리에서 크롤링 등으로 얻어낸 코퍼스 데이터가 필요에 맞게 전처리되지 않은 상태라면, 해당 데이터를 사용하고자하는 용도에 맞게 토큰화(tokenization) & 정제(cleaning) & 정규화(normalization)하는 일을 하게 된다.

> 토큰화 (tokenization)는 주어진 코퍼스(corpus)에서 토큰(token)이라고 불리는 단위로 나누는 작업을 말한다. 보통 의미있는 단위로 토큰을 정의한다.

## 1. 단어 토큰화 (Word Tokenization)

> 토큰의 기준이 단어(Word)

- 아래의 입력으로부터 구두점(punctuation)과 같은 문자는 제외시키는 토큰화작업을 해보자.
  - 구두점이란 마침표, 컴마, 물음표, 세미콜론, 느낌표 등과 같은 기호를 말한다.
- Time is an illusion. Lunchtime double so!

  → "Time", "is", "an", "illustion", "Lunchtime", "double", "so"

  해당 작업은 간단하다. 구두점을 지운 뒤에 띄어쓰기(whitespace)를 기준으로 잘라냈다.

- 보통 토큰화 작업은 단순히 구두점이나 특수문자를 전부 제거하는 정제(cleaning) 작업을 수행하는 것만으로 해결되지는 않는다. 구두점이나 특수문자를 전부 제거하면 토큰이 의미를 잃어버리는 경우가 발생하기도 한다.

## 2. 토큰화 중 생기는 선택의 순간

- 영어권 언어에서 아포스트로피(')가 들어가 있는 단어는 어떻게 토큰으로 분류해야할까라는 문제를 보자
- Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.

  → Don't, Jone's는 어떻게 토큰화를 해야할까?

- NLTK는 영어 코퍼스를 토큰화하기 위한 도구들을 제공한다. 그 중 work_tokenize와 WordPunctTokenizer를 사용해서 NLTK에서는 아포스트로피를 어떻게 처리하는지 확인해보자.

```python
from nltk.tokenize import word_tokenize

print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

'''
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
'''
```

```python
from nltk.tokenize import WordPunctTokenizer
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

'''
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
'''
```

- WordPunctTokenizer는 구두점을 별도로 분류하는 특징을 갖고 있기때문에, 앞서 확인했던 word_tokenize와는 달리 Don't를 Don과 '와 t로 분리하였다.
- 케라스 또한 토큰화 도구로서 text_to_word_sequence를 지원한다.

```python
from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

'''
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
'''
```

- 케라스의 text_to_word_sequence는 기본적으로 모든 알파벳을 소문자로 바꾸면서 마침표나 컴마, 느낌표 등의 구두점을 제거하지만 don't나 jone's와 같은 경우 아포스트로피는 보존하는 것을 볼 수 있다.

## 3. 토큰화에서 고려해야할 사항

> 구두점이나 특수 문자를 단순 제외해서는 안된다.

> 줄임말과 단어 내에 띄어쓰기가 있는 경우.

- what're는 what are의 줄임말이며, we're는 we are의 줄임말이다. 여기서 쓰인 re를 접어(clitic)라고 부른다. 즉, 단어가 줄임말로 쓰일 때 생기는 형태를 말한다.
- New York라는 단어나 rock'n'roll 이라는 단어를 보면 이 단어들은 하나의 단어이지만 중간에 띄어쓰기가 존재한다. 사용 용도에 따라서, 하나의 단어 사이에 띄어쓰기가 있는 경우에도 하나의 토큰으로 봐야하는 경우도 있을 수 있으므로, 토큰화 작업은 이러한 단어를 하나로 인식할 수 있는 능력도 가져야 한다.

> 표준 토큰화 예제

- 표준 토큰화 방법 중 하나인 Penn Treebank Tokenization의 규칙에 대해서 소개한다.
  1. 하이픈으로 구성된 단어는 하나로 유지한다.
  2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

```python
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))

'''
['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
'''
```

- 'home-based '는 하나의 토큰으로 취급, doesn't는 does와 n't로 분리됨

## 4. 문장 토큰화 (Sentence Tokenization)

> 토큰의 단위가 문장(sentence)

- 코퍼스 내에서 문장 단위로 구분하는 작업으로 때로는 문장 분류(sentence segmentation)라고도 부른다.
- 어떻게 주어진 코퍼스로부터 문장 단위로 분류할 수 있을까? 직관적으로 생각해봤을 때는 ?난 마침표(.)나 ! 기준으로 문자을 잘라내면 되지 않을까라고 생각할 수 있지만, 꼭 그렇지만은 않다.

  → !나 ?는 문장의 구분을 위한 꽤 명확한 구분자(boundary)역할을 하지만 마침표는 꼭 그렇지 않기 때문이다.

**EX1) IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 ukairia777@gmail.com로 결과 좀 보내줘. 그러고나서 점심 먹으러 가자.**

**EX2) Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.**

- 우의 예제들에 마침표를 기준으로 문장 토큰화를 적용해본다면, 첫번째 예제에서는 보내줘. 두번째 예제에서는 year. 에서 처음으로 문장이 끝난 것으로 인식하느 것이 제대로 문장의 끝을 예측했다고 볼 수 있을 것 인데, 마침표를 기준으로 실제로 적용하면 이미 여러번 등장해서 예상한 결과가 나오지 않는다.
- 그렇기 때문에 사용하는 코퍼스가 어떤 국적의 언어인지, 또는 해당 코퍼느 내에서 특수문자들이 어떻게 사용되고 있는지에 따라 직접 규칙들을 정의해볼 수 있을 것 이다. 물론, 100% 정확도를 얻는 일은 쉬운 일이 아니지만 말이다.
- NLTK 에서는 영어 문장의 토큰화를 수행하는 sent_tokenize를 지원하고 있다.

```python
from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))
'''
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
'''
```

- 마침표로 구분해내고 있는 것을 볼 수 있다. 위의 예제도 한번 확인해보자.

```python
from nltk.tokenize import sent_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
'''
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
'''
```

- NLTK는 단순히 마침표를 구분자로 하여 문장을 구분하지 않았기 때문에, Ph.D를 문장 내의 단어로 인식하여 성공적으로 인식하는 것을 볼 수 있다.
- 한국어에 대한 문장 토큰화 도구 또한 존재한다. KSS(Korean Sentence Splitter) 이다.

```python
import kss

text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))

'''
['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '농담아니에요.', '이제 해보면 알걸요?']
'''
```

## 5. 이진 분류기 (Binary Classfier)

- 문장 토큰화에서는 예외 사항을 발생시키는 마침표의 처리를 위해서 입력에 따라 2개의 클래스로 분류하는 이진 분류기(binary classifier)를 사용하기도 한다.
- 마침표에 대한 두 개의 클래스
  1. 마침표(.)가 단어의 일부분일 경우. 즉, 마침표가 약어(abbreivation)로 쓰이는 경우
  2. 마침표(.)가 정말로 문장의 구분자(boundary)일 경우

## 6. 한국어 토큰화의 어려움

- 한국어의 경우에는 띄어쓰기가 단위가 되는 단위를 '어절'이라고 하는데, 어절 토큰화는 한국어 NLP에서 지양되고 있다. 어절 토큰화와 단어 토큰화가 같지 않기 때문이다.

> 한국어는 교착어이다.

- 한국어에는 다양한 조사라는 것이 존재한다. 같은 단어가 와도 서로 다른 조사가 붙어서 다른 단어로 인식이 되면 자연어 처리가 힘들고 번거로워지는 경우도 많다. 대부분의 한국어 NLP에서 조사는 분리해줄 필요가 있다.
- 한국어 토큰화에서는 형태소(morpheme)란 개념이 존재한다. 형태소란 뜻을 가진 가장 작은 말의 단위를 말한다. 이에는 자립형태소와 의존형태소가 존재한다.
- 한국어에서 영어에서의 단어 토큰화와 유사한 형태를 얻으려면 어절 토큰화가 아니라, 형태로 토큰화를 수행해야 한다.

> 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않는다.

- 가장 기본적인 견해는 한국어의 경우, 띄어쓰기가 지켜지지 않아도 글을 쉽게 이해할 수 있는 언어라는 점이다. 띄어쓰기가 보편화된 것도 근대, 1933년의 일이다.
- 하지만 영어는 띄어쓰기를 하지 않으면 손쉽게 알아보기 어려운 문장들이 생긴다.
- 한국어(모아쓰기 방식)와 영어(풀어쓰기 방식)라는 언어적 특성의 차이에 기인한다.

## 7. 품사 태깅 (part-of-speech tagging)

- 단어는 표기는 같지만, 품사에 따라서 의미가 달라지기도 한다.
- 단어의 의미를 제대로 파악하기 위해서는 해당 단어가 어떤 품사로 쓰였는지 보는 것이 주요 지표가 될 수 있는데, 그에 따라 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해놓기도 하는데, 이를 품사 태깅이라고 한다.

## 8. NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습

- NLTK에서는 Penn Treebank POS Tags라는 기준을 사용한다.

```python
from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))
'''
['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
'''

from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)
'''
[('I', 'PRP'),
 ('am', 'VBP'),
 ('actively', 'RB'),
 ('looking', 'VBG'),
 ('for', 'IN'),
 ('Ph.D.', 'NNP'),
 ('students', 'NNS'),
 ('.', '.'),
 ('and', 'CC'),
 ('you', 'PRP'),
 ('are', 'VBP'),
 ('a', 'DT'),
 ('Ph.D.', 'NNP'),
 ('student', 'NN'),
 ('.', '.')]
'''
```

- 영어 문장에 대해서 토큰화를 수행하고, 이어서 품사 태깅을 수행했다. Penn Treebank POG Tags에서 PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사 등등등을 의미한다.
- 한국어 자연어 처리를 위해서는 KoNLPy라는 파이썬 패키지를 사용할 수 있다.
- 한국어 NLP에서 형태소 분석기를 사용한다는 것은 단어 토큰화가 아니라, 정확히는 형태로 단위로 형태소 토큰화를 수행하게 됨을 뜻한다.

```python
from konlpy.tag import Okt
okt=Okt()
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
'''
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
[('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
'''
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
['코딩', '당신', '연휴', '여행']
'''
```

1. morphs: 형태소 추출
2. pos: 품사 태깅
3. nouns: 명사 추출

```python
from konlpy.tag import Kkma
kkma=Kkma()
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
'''
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
[('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
'''
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
['코딩', '당신', '연휴', '여행']
'''
```

- 각 형태소 분석기는 성능과 결과가 다르게 나오기 때문에, 형태소 분석기의 선택은 사용하고자 하는 필요 용도에 어떤 형태소 분석기가 가장 적절한지를 판단하고 사용하면 된다.

# 2. 정제(Cleaning) and 정규화(Normalization)

- 코퍼스에서 용도에 맞게 토큰을 분류하는 작업을 토큰화라고 하며, 토큰화 작업 전, 후에는 텍스트 데이터를 용도에 맞게 정제(cleaning) 및 정규화(normalization)하는 일이 항상 함께한다.

> 정제(cleaning) : 갖고 있는 코퍼스로부터 노이즈 데이터를 제거한다.

> 정규화(normalization) : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어준다.

- 토큰화 작업 이후에도 여전히 남아있는 노이즈들을 제거하기 위해 지속적으로 정제작업을 하기도 한다. 완벽한 정제 작업은 어려운 편이라, 대부분의 경우 이 정도면 됐다.라는 일종의 합의점을 찾기도 한다.

### 1. 규칙에 기반한 표기가 다른 단어들의 통합

- 표기가 다른 단어들(뜻은 같음)을 통합하는 방법에는 어간 추출(stemming)과 표제어 추출(lemmatization)이 있다.

### 2. 대, 소문자 통합

- 영어권 언어에서 대,소문자를 통합하는 것은 단어의 개수를 줄일 수 있는 또 다른 정규화 방법이다. 영어권 언어에서 대문자는 문장의 맨 앞등과 같은 특정 상황에서만 쓰이고, 대부분의 글은 소문자로 작성되기 때문에 대, 소문자 작업은 대부분 대문자를 소문자로 변환하는 소문자 변환작업으로 이루어지게 된다.
- 모든 토큰을 소문자로 만드는 것이 문제를 가져온다면, 또 다른 대안은 일부만 소문자로 변환시키는 방법도 있다.
- 이러한 작업은 더 많은 변수를 사용해서 소문자 변환을 언제 사용할지 결정하는 머신러닝 시퀀스 모델로 더 정확하게 진행시킬 수 있다.

### 3. 불필요한 단어의 제거 (Removing Unnecessary Words)

- 정제 작업에서 제거해야하는 노이즈 데이터(noise data)는 자연어가 아니면서 아무 의미도 갖지 않는 글자들을 의미하기도 하지만, 분석하고자 하는 목적에 맞지 않는 불필요 단어들을 노이즈 데이터라고 하기도 한다.
- 불필요 단어들을 제거하는 방법으로는 불용어 제거와 등장 빈도가 적은 단어, 길이가 짧은 단어들을 제거하는 방법이 있다.

> 등장 빈도가 적은 단어 (Removing Rare words)

- 스팸 메일을 구분하는 스팸 메일 분류기를 설계할 때, 총 100,000개의 메일을 가지고 정상 메일에서는 어떤 단어들이 주로 등장하고, 스팸 메일에서는 어떤 단어들이 주로 등장하는지를 가지고 설계하고자 한다.
- 이 때 100,000개의 메일 데이터에서 총 합 5번 밖에 등장하지 않은 단어가 있다면 이 단어는 직관적으로 분류에 거의 도움이 되지 않을 것임을 확인할 수 있다.

> 길이가 짧은 단어 (Removing words with a very short length)

- 영어권 언어에서는 길이가 짧은 단어를 삭제하는 것만으로도 어느정도 자연어 처리에서 크게 의미가 없는 단어들을 제거하는 효과를 볼 수 있다고 알려져 있다.
- 즉, 영어권 언어에서 길이가 짧은 단어들은 대부분 불용어에 해당된다. 사실 길이가 짧은 단어를 제거하는 2차 이유는 길이를 조건으로 텍스트를 삭제하면서 단어가 아닌 구두점들까지도 한꺼번에 제거하기 위함도 있다.
- 단정적으로 말할 수는 없지만, 영어 단어의 평균 길이는 6~7 정도이며, 한국어 단어의 평균 길이는 2~3 정도로 추정되고 있다.
- 이는 영어 단어와 한국어 단어에서 각 한 글자가 가진 의미의 크기가 다르다는 점에서 기인한다. 한국어 단어는 한자어가 많고, 한 글자만으로도 이미 의미를 가진 경우가 많다.
- 이러한 특성으로 인해 영어는 길이가 2~3 이하인 단어를 제거하는 것만으로도 크게 의미를 갖지 못하는 단어를 줄이는 효과를 갖고 있다.
- 하지만 fox, dog, car 등 길이가 3인 명사들이 제거 되기 시작하므로, 사용하고자 하는 데이터에서 해당 방법을 사용해도 되는지에 대한 고민이 필요하다.

```python
import re
text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))

# was wondering anyone out there could enlighten this car.
# I, if, me, on 제거
# was, out, car 등은 제거 되지 않음
```

# 3. 어간 추출(Stemming) and 표제어 추출(Lemmatization)

- 이 두 작업이 가지고 있는 의미는 눈으로 봤을 때는 서로 다른 단어들이지만, 하나의 단어로 일반화시킬 수 있다면 하나의 단어로 일반화시켜서 문서 내의 단어 수를 줄이겠다는 것이다.
- 이러한 방법들은 단어의 빈도수를 기반으로 문제를 풀고자 하는 BoW(Bag of Words)표현을 사용하는 자연어 처리 문제에서 주로 사용된다.
- **자연어 처리에서 전처리, 더 정확히 정규화의 지향점은 언제나 갖고 있는 코퍼스로부터 복잡성을 줄이는 일이다.**

## 1. 표제어 추출 (Lemmatization)

- 표제어(Lemma) 추출은 단어들로부터 표제어를 찾아가는 과정이다.
- 표제어 추출은 단어들이 다른 형태를 가지더라도, 그 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단한다.

  → am, are, is는 서로 다른 스펠링이지만, 그 뿌리 단어는 be

- 표제어 추출을 하는 가장 섬세한 방법은 단어의 형태학적 파싱을 먼저 진행하는 것인데, 형태소란 의미를 가진 가장 작은 단위를 뜻한다. 그리고 형태학(morphology)이란, 형태소로부터 단어들을 만들어가는 학문을 뜻한다.
- 어간(stem)

  단어의 의미를 담고 있는 단어의 핵심 부분.

- 접사 (affix)

  단어에 추가적인 의미를 주는 부분

```python
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([n.lemmatize(w) for w in words])

# ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
```

- 표제어 추출은 어간 추출과는 달리 단어의 형태가 적절히 보존되는 양상을 보이는 특징이 있다. 하지만 그럼에도 dy나 ha와 같이 의미를 알 수 없는 적절하지 못한 단어를 출력하고 있다.
- 이는 표제어 추출기(lemmatizer)가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문이다.
- WordNetLemmatizer는 입력으로 단어가 동사 품사라는 사실을 알려줄 수 있다. 즉, dies와 watched, has가 문장에서 동사로 쓰였다는 것을 알려준다면 표제어 추출기는 품사의 정보를 보존하면서 정확한 Lemma를 출력하게 된다.

```python
n.lemmatize('dies', 'v') # die
```

- 이렇듯 표제어 추출은 문맥을 고려하며, 수행했을 때의 결과는 해당 단어의 품사 정보를 보존하지만, 어간 추출을 수행한 결과는 품사 정보가 보존되지 않는다.

## 2. 어간 추출 (Stemming)

- 어간(Stem)을 추출하는 작업을 어간 추출(stemming)이라고 한다. 어간 추출은 형태학적 분석을 단수화한 버전이라고 볼 수도 있고, 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업이라고도 볼 수 있다.

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)

print([s.stem(w) for w in words])

# ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']

# ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
```

- 위의 어간 추출 알고리즘의 포터 알고리즘은 단순 규칙에 기반하여 이루어지기 때문에 사전에 없는 단어들이 포함될 수도 있다.

  ALIZE → AL , ANCE → 제거, ICAL → IC

- 어간 추출의 장점은 속도면에서 표제어 추출보다 일반적으로 빠르다는 점이다. 여기서 포터 어간 추출기는 정밀하게 설계되어 정확도가 높으므로, 영어 자연어 처리에서 어간 추출을 하고자 한다면 가장 준수한 선택이다.

## 3. 한국어에서의 어간 추출

- 한국어는 5언 9품사의 구조를 가지고 있다.
  - 체언 : 명사, 대명사, 수사
  - 수식언 : 관형사, 부사
  - 관계언 : 조사
  - 독립언 : 감탄사
  - 용언 : 동사, 형용사

> **활용 (conjugation )**

- 활용이란 용언의 어간이 어미를 가지는 일을 말한다.

> 규칙 활용

- 규칙 활용은 어간이 어미를 취할 때, 어간의 모습이 일정하다.

> 불규칙 활용

- 뷸규칙 활용은 어간이 어미를 취할 때, 어간의 모습이 바뀌거나 취하는 어미가 특수한 어미일 경우를 말한다.

# 4. 불용어 (Stopword)

- 갖고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업이 필요하다.
- 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하는 바가 없는 경우의 단어들을 불용어(stopword)라고 한다.
- nltk 에서 지원해주는 불용어 리스트

  ```python
  from nltk.corpus import stopwords
  stopwords.words('english')[:10]
  ```

- nltk를 통해서 불용어 제거하기

  ```python
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize

  example = "Family is not an important thing. It's everything."
  stop_words = set(stopwords.words('english'))

  word_tokens = word_tokenize(example)

  result = []
  for w in word_tokens:
      if w not in stop_words:
          result.append(w)

  print(word_tokens)
  # ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
  print(result)
  # ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
  ```

# 5. 정규 표현식 (Regular Expression)

- 정규 표현식을 이용한 토큰화

```python
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

# ['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

# [\w]+ 는 한개 이상의 단어또는 숫자를 의미,
```

```python
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

# ["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
# [\s]+ 는 공백을 뜻하는데, gaps 파라미터를 넘겨 해당 단어 매칭 토크나이저가 아닌, 정규표현식으로 구분하는 토크나이저를 만든다.
```

# 6. 정수 인코딩 (Integer Encoding)

- 컴퓨터는 텍스트보다는 숫자를 더 잘 처리할 수 있다.
- 이렇듯 인덱스를 각 단어들에게 부여하는데, 인덱스를 부여하는 방법은 여러 가지가 있을 수 있다. 보통은 전처리 또는 빈도수가 높은 단어들만 사용하기 위해서 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여한다.

## 1. 정수 인코딩 (Integer Encoding)

- 단어에 정수를 부여하는 방법 중 하나로, 단어를 빈도수 순으로 정렬한 단어 집합(vocabulary)을 만들고, 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수를 부여하는 방법이 있다.

> dictionary 사용하기

```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
text = sent_tokenize(text)
print(text)

# 정제와 단어 토큰화
vocab = {} # 파이썬의 dictionary 자료형
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence:
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    sentences.append(result)
print(sentences)
print(vocab)
# {'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}
```

- 텍스트를 숫자로 바꾸는 단계라는 것은 본격적으로 자연어 처리 작업에 들어간다는 의미이므로, 단어가 텍스트일 때만 할 수 있는 최대한의 전처리를 끝내놓아야 한다.

```python
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)
# [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]
```

```python
word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word] = i
print(word_to_index)
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
```

- 이렇게 텍스트 빈도수에 따른 인덱스 번호 부여를 완료했다. 하지만 이 단어들을 모두 사용하기 보다는 빈도수가 가장 높은 n개의 단어만 사용하고 싶은 경우가 많다.
- 또한 단어 집합에 존재하지 않는 단어들을 Out-Of-Vocabulary의 약자로 OOV로 부르고, 단어 집합에 없는 단어들은 OOV로 인코딩 하도록 한다.

```python
vocab_size = 5
words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
print(word_to_index)

word_to_index['OOV'] = len(word_to_index) + 1

encoded = []
for s in sentences:
    temp = []
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)

# [[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]

```

> Counter 이용하기

```python
from collections import Counter

words = sum(sentences, [])
# 위 작업은 words = np.hstack(sentences)로도 수행 가능.
print(words)
# ['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'huge', 'secret', 'huge', 'secret', 'barber', 'kept', 'word', 'barber', 'kept', 'word', 'barber', 'kept', 'secret', 'keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy', 'barber', 'went', 'huge', 'mountain']

vocab = Counter(words) # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.
print(vocab)
# Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
vocab
# [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]

word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i+1
    word_to_index[word] = i
print(word_to_index)
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

> NLTK의 FreqDist 사용하기

## 2. 케라스(Keras)의 텍스트 전처리

- jupyter notebook 확인
- 케라스 토크나이저는 기본적으로 단어 집합에 없는 단어인 OOV에 대해서는 단어를 정수로 바꾸는 과정에서 아예 단어를 제거한다는 특징을 가지고 있다.
- 단어 집합에 없는 단어들은 OOV로 간주하여 보존하고 싶다면, Tokenizer의 인자 oov_token을 사용한다.

# 7. 패딩 (padding)

- 자연어 처리를 하다보면 각 문장(또는 문서)은 서로 길이가 다를 수 있다. 그런데 기계는 길이가 전부 동일한 문서들에 대해서는 하나의 행렬로 보고, 한꺼번에 묶어서 처리할 수 있다. 다시 말해 병렬 연산을 위해서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요할 때가 있다.

## 1. Numpy 패딩

```python
sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# Text To Sequence
[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```

- 동일한 길이로 맞춰주기 위해서 가장 길이가 긴 문장의 길이를 계산해보자.

```python
max_len = max(len(item) for item in encoded)
print(max_len) # 7
```

- 가장 길이가 긴 문장의 길이는 7인데, 모든 문장의 길이를 7로 맞추어 줄 것 이다. 이 때 가상의 단어 'PAD'를 사용한다. 그리고 해당 단어를 0번 단어라고 정의해보자

```python
for item in encoded: # 각 문장에 대해서
    while len(item) < max_len:   # max_len보다 작으면
        item.append(0)

padded_np = np.array(encoded)
padded_np
'''
array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]])
'''
```

- 길이가 7보다 짧은 문장에는 전부 숫자 0이 뒤로 붙어서 모든 문장의 길이가 전부 7이 된 것을 알 수 있다. 기계는 이제 이들을 하나의 행렬로 보고, 병렬 처리를 할 수 있다.
- 또한, 0번 단어는 사실 아무런 의미도 없는 단어이기 때문에 자연어 처리 과정에서 기계는 0번 단어를 무시하게 될 것 이다.
- 이와 같이 특정 값을 채워서 데이터의 크기(shape)를 조정하는 것을 패딩(padding)이라고 한다.
- 숫자 0을 사용하고 있다면 제로 패딩(zero padding)이라고 한다.

## 2. 케라스 전처리 도구로 패딩하기

- jupyter notebook "Padding" 참고
- Numpy 패딩 결과와는 다른 것을 확인할 수 있는데, 이는 pad_sequences는 기본적으로 문서의 뒤에 0을 채우는 것이 아니라, 앞에 0으로 채우기 때문이다.
- 뒤에 0을 채우고 싶다면 인자로 padding='post'를 주면된다.
- 실제로는 꼭 가장 긴 문서의 길이를 기준으로 해야하는 것은 아니다. 가령, 모든 문서의 평균 길이가 20인데 문서 1개의 길이가 5000이라고 해서 굳이 모든 문서의 길이를 5000으로 패딩할 필요는 없을 수 있다. 이 때는 max_len 인자를 보내주면 된다.

  → 길이가 5보다 짧은 문서들은 0으로 패딩되고, 기존에 5보다 길었다면 데이터가 손실된다.

# 8. 원-핫 인코딩 (One-Hot Encoding)

- 컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리할 수 있다. 이를 위해 자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법들이 있다.
- 원-핫 인코딩은 그 많은 기법 중에서 단어를 표현하는 가장 기본적인 표현 방법이며, 머신 러닝, 딥 러닝을 하기 위해서는 반드시 배워야 하는 표현 방법이다.

## 1. 원-핫 인코딩(One-Hot Encoding)이란?

- 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다.
- 이렇게 표현된 벡터를 원-핫 벡터 (One-Hot vector)라고 한다.

```python
# Tokenizing
from konlpy.tag import Okt
okt=Okt()
token=okt.morphs("나는 자연어 처리를 배운다")
print(token)

# Word To Sequences
word2index={}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca]=len(word2index)
print(word2index)

# Function one_hot_encoding
def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index=word2index[word]
       one_hot_vector[index]=1
       return one_hot_vector

one_hot_encoding("자연어",word2index)
# [0, 0, 1, 0, 0, 0]
```

- "자연어" 라는 단어를 표현하고자 하는 벡터의 집합에서는 인덱스 2의 값이 1이며, 나머지 값은 0인 벡터가 나온다.

## 2. 케라스(Keras)를 이용한 원-핫 인코딩(One-Hot Encoding)

- 위에서는 직접 함수로 구현을 했지만, 케라스는 원-핫 인코딩을 수행하는 유용한 도구 to_categorical을 지원한다.

## 3. 원-핫 인코딩(One-Hot Encoding)의 한계

- 이러한 표현 방식은 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다는 단점이 있다. 다른 말로는 벡터의 차원이 계속 늘어난다고도 표현한다.
- 원-핫 벡터는 단어의 유사도를 표현하지 못한다는 단점이 있다.
- 이러한 단점을 해결하기 위해 단어의 잠재 의미를 반영하여 다차원 공간에 벡터화 하는 기법으로 크게 두 가지가 있다. 첫째는 카운트 기반의 벡터화 방법인 LSA, HAL 등이 있으며, 둘째는 예측 기반으로 벡터화하는 NNLM, RNNLM, Word2Vec, FastText 등이 있다.
- 그리고 카운트기반과 예측기반 두 가지 방법을 모두 사용하는 방법으로 GloVe라는 방법이 존재한다.

# 9. 데이터의 분리 (Splitting Data)

- 머신 러닝(딥 러닝) 모델에 데이터를 훈련시키기 위해서는 데이터를 적절히 분리하는 작업이 필요하다.

## 1. 지도 학습 (Supervised Learning)

- 지도 학습의 훈련 데이터는 문제지를 연상케 하는데, 지도 학습의 훈련 데이터는 정답이 무엇인지 맞춰야 하는 '문제'에 해당되는 데이터와 레이블이라고 부르는 '정답'이 적혀 있는 데이터로 구성되어 있다.
- 기계는 정답이 적혀져 있는 문제지를 문제와 정답을 함께 보면서 열심히 공부하고, 향후에 정답이 없는 문제에 대해서도 정답을 잘 예측해야 한다.
- 기계를 가르치기 위해서는 데이터를 총 4개로 나눈다. 우선 메일의 내용이 담긴 첫 번째 열을 X에 저장한다. 그리고 메일이 스팸인지 정상인지 정답이 적혀있는 두번째 열을 y에 저장한다.
- 다시 이 x와 y에 대해서 일부 데이터를 또 다시 분리하여 시험용으로 일부 문제와 정답지를 빼놓는 것이다.
- 훈련 데이터

  X_train : 문제지 데이터

  y_train : 문제지에 대한 정답 데이터

- 테스트 데이터

  X_test : 시험지 데이터

  y_test : 시험지에 대한 정답 데이터

- 기계가 이러한 변수들을 부여받으면, X_train과 y_train에 대해서 학습을 한다. 기계는 현 상태에서는 정답지인 y_train을 볼 수 있기 때문에 18,000개의 문제지 X_train을 보면서 어떤 메일 내용일 때, 정상 메일인지 스팸 메일인지를 열심히 규칙을 도출해나가면서 정리해나간다.
- 그리고 학습을 다 한 기계에게 y_test는 보여주지 않고, X_test에 대해서 정답을 예측하게 한다.
- 그리고 기계가 예측한 답과 실제 정답인 y_test를 비교하면서 기계가 정답을 얼마나 맞췄는지를 평가한다.
- 이 수치가 기계의 정확도(Accuracy)가 된다.

## 2. X와 y분리하기

### 1. zip 함수를 이용하여 분리하기

- zip() 함수는 동일한 개수를 가지는 시퀀스 자료형에서 각 순서에 등장하는 원소들끼리 묶어주는 역할을 한다.

### 2. 데이터프레임을 이용하여 분리하기

### 3. Numpy를 이용하여 분리하기

## 3. 테스트 데이터 분리하기

### 1. 사이킷런을 이용하여 분리하기

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
```

- X : 독립 변수 데이터 (배열이나 데이터프레임)
- y: 종속 변수 데이터. 레이블 데이터.
- test_size: 테스트용 데이터 개수를 지정한다.
- train_size: 학습용 데이터의 개수를 지정한다.
- random_state : 난수 시드

```python
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
print(X)
'''
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
'''
print(list(y)) #레이블 데이터
# [0, 1, 2, 3, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
#3분의 1만 test 데이터로 지정.
#random_state 지정으로 인해 순서가 섞인 채로 훈련 데이터와 테스트 데이터가 나눠진다.

print(X_train)
print(X_test)
'''
[[2 3]
 [4 5]
 [6 7]]
[[8 9]
 [0 1]]
'''

print(y_train)
print(y_test)
'''
[1, 2, 3]
[4, 0]
'''
```

### 2. 수동으로 분리하기

- jupyter notebook 참고

# 10. 한국어 전처리 패키지 (Text Preprocessing Tools for Korean Text)

## 1. PyKoSpacing

- 띄어쓰기가 되어 있지 않은 문장을 띄어쓰기를 한 문장으로 변환해주는 패키지

## 2. Py-Hanspell

- 네이버 한글 맞춤법 검사기를 바탕으로 만들어진 패키지이다.
- 띄어쓰기 또한 보정한다.

## 3. SOYNLP를 이용한 단어 토큰화

- 품사 태깅, 단어 토큰화 등을 지원하는 단어 토크나이저이다.
- 비지도 학습으로 단어 토큰화를 한다는 특징을 갖고 있고, 데이터에 자주 등장하는 단어들을 단어로 분석한다.

### 1. 신조어 문제

```python
from konlpy.tag import Okt
tokenizer = Okt()
print(tokenizer.morphs('에이비식스 이대휘 1월 최애돌 기부 요정'))
# ['에이', '비식스', '이대', '휘', '1월', '최애', '돌', '기부', '요정']
```

### 2. 학습하기

- soynlp는 기본적으로 학습에 기반한 토크나이저이므로, 학습에 필요한 한국어 문서를 다운로드한다.

```python
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
```

- solynlp는 학습 기반의 단어 토크나이저이므로, 기존의 KoNLPy에서 제공하는 형태소 분석기들과는 달리 학습 과정을 거쳐야 한다. 이는 전체 코퍼스로부터 응집 확률과 브랜칭 엔트로피 단어 점수표를 만드는 과정이다.

```python
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
```

### 3. SOYNLP의 응집 확률 (cohesion probability)

- 응집 확률은 내부 문자열(substring)이 얼마나 자주 등장하는지를 판다하는 척도이다.
- 응집 확률은 문자열을 문자 단위로 분리하여, 내부 문자열을 만드는 과정에서 왼쪽부터 순서대로 문자를 추가하면서 각 문자열이 주어졌을 때 그 다음 문자가 나올 확률을 계산하여 누적곱을 한 값이다. 이 값이 높을 수록 전체 코퍼스에서 이 문자열 시퀀스는 하나의 단어로 등장할 가능성이 높다.

$$cohesion(2) = P(반포|반)$$

$$cohesion(3) = \sqrt{P(반포|반)\cdot P(반포한|반포))}^2$$

$$cohesion(4) = \sqrt{P(반포|반)\cdot P(반포한|반포) \cdot P(반포한강|반포한))}^3$$

```python
word_score_table["반포한"].cohesion_forward # 0.08838002913645132
word_score_table["반포한강"].cohesion_forward # 0.19841268168224552
word_score_table["반포한강공"].cohesion_forward # 0.2972877884078849
word_score_table["반포한강공원"].cohesion_forward # 0.37891487632839754
word_score_table["반포한강공원에"].cohesion_forward # 0.33492963377557666
```

### 4. SOYNLP의 브랜칭 엔트로피(branching entropy)

- Branching Entropy는 확률 분포의 엔트로피값을 사용한다. 이는 주어진 문자열에서 얼마나 다음 문자가 등장할 수 있는지를 판단하는 척도이다.
- Branching Entropy를 주어진 문자 시퀀스에서 다음 문자 예측을 위해 헷갈리는 정도로 비유를 해보자.
- Branching Entropy의 값은 하나의 완성된 단어에 가까워질수록 문맥으로 인해 점점 정확히 예측할 수 있게 되면서 점점 줄어드는 양상을 보인다.

```python
word_score_table["디스"].right_branching_entropy # 1.6371694761537934
word_score_table["디스플"].right_branching_entropy # -0.0
word_score_table["디스플레"].right_branching_entropy # -0.0
word_score_table["디스플레이"].right_branching_entropy # 3.1400392861792916
```

- 디스 → 플 → 레 까지는 너무 명백하기 때문에 0이라는 값을 가진다. 하지만 디스플레이 이후에는 갑자기 값이 증가한다.
- 그 이유는 문자 시퀀스 '디스플레이'라는 문자 시퀀스 다음에는 조사나 다른 단어와 같은 다양한 경우가 있을 수 있기 때문이다.

### 5. SOYNLP의 L tokenizer

- 한국어는 띄어쓰기 단위로 나눈 어절 토큰은 주로 L 토큰 + R 토큰의 형식을 가질 때가 많다.
- L 토크나이저는 L 토큰 + R 토큰으로 나누되, 분리 기준을 점수가 가장 높은 L 토큰을 찾아내는 원리를 가지고 있다.

```python
from soynlp.tokenizer import LTokenizer

scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
l_tokenizer = LTokenizer(scores=scores)
l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False)
# [('국제사회', '와'), ('우리', '의'), ('노력', '들로'), ('범죄', '를'), ('척결', '하자')]
```

### 6. 최대 점수 토크나이저

- 최대 점수 토크나이저는 띄어쓰기가 되지 않는 문장에서 점수가 높은 글자 시퀀스를 순차적으로 찾아내는 토크나이저이다.

```python
from soynlp.tokenizer import MaxScoreTokenizer

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
maxscore_tokenizer.tokenize("국제사회와우리의노력들로범죄를척결하자")
# ['국제사회', '와', '우리', '의', '노력', '들로', '범죄', '를', '척결', '하자']
```

## 4. SOYNLP를 이용한 반복되는 문자 정제

```python
from soynlp.normalizer import *

print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))

'''
아ㅋㅋ영화존잼쓰ㅠㅠ
아ㅋㅋ영화존잼쓰ㅠㅠ
아ㅋㅋ영화존잼쓰ㅠㅠ
아ㅋㅋ영화존잼쓰ㅠㅠ
'''
```

# 5. Customized KoNLPy

- 영어권 언어는 띄어쓰기만해도 단어들이 잘 분리되지만, 한국어는 그렇지 않다고 몇 차례 언급되었다.
- 한국어 데이터를 사용하여 모델을 구현하는 것만큼 형태소 분석기를 사용해서 단어 토큰화를 해보자

```
형태소 분석 입력 : '은경이는 사무실로 갔습니다.'
형태소 분석 결과 : ['은', '경이', '는', '사무실', '로', '갔습니다', '.']
```

- 위와 같은 경우, 은경이는 하나의 단어라는 것을 형태소 분석기에 알려줘야 한다.

```python
from ckonlpy.tag import Twitter
twitter = Twitter()
twitter.morphs('은경이는 사무실로 갔습니다.')

twitter.add_dictionary('은경이', 'Noun')

twitter.morphs('은경이는 사무실로 갔습니다.')

# ['은경이', '는', '사무실', '로', '갔습니다', '.']
```
