# 순환 신경망 (Recurrent Neural Network)

- 피드 포워드 신경망은 입력의 길이가 고정되어 있어, 자연어 처리를 위한 신경망으로는 한계가 있었다. 결국 다양한 길이의 입력 시퀀스를 처리할 수 있는 인공 신경망이 필요하게 되었는데, 자연어 처리에 대표적으로 사용되는 인공신경망인 RNN, LSTM 등에 대해 배워보자.

# 1. 순환 신경망 (Recurrent Neural Network, RNN)

- RNN은 시퀀스(Sequence) 모델이다. 입력과 출력을 시퀀스 단위로 처리하는 모델인데, 번역기를 생각해보면 입력은 번역하고자 하는 문장. 즉, 단어 시퀀스이다.
- 출력에 해당되는 번역된 문장 또한 단어 시퀀스이다.
- 이렇듯 시퀀스들을 처리하기 위해 고안된 모델들을 시퀀스 모델이라고 한다. 그 중에서도 RNN은 딥 러닝에 있어, 가장 기본적인 시퀀스 모델이다.

## 1. 순환 신경망 (Recurrent Neural Network, RNN)

- 앞서 배운 신경망들은 전부 은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향했다. 이와 같은 신경망들을 피드 포워드 신경망이라고 한다.
- RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서 다시 은닉층 노드의 다음 계싼의 입력으로 보내는 특징을 갖고 있다.
- RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 셀(cell)이라고 한다.

  → 이 셀은 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행하므로, 이를 메모리 셀 또는 RNN 셀이라고 표현한다.

- 은닉층의 메모리 셀은 각각의 시점(time step)에서 바로 이전 시점에서의 은닉층의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동을 하고 있다.

  → 현재 시점 t에서의 메모리 셀이 갖고 있는 값은 과거의 메모리 셀들의 값에 영향을 받은 것임을 의미한다.

- 메모리 셀이 출력층 방향으로 또는 다음 시점 t+1의 자신에게 보내는 값을 은닉상태(hidden state)라고 한다. 다시 말해 t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 은닉 상태값을 t 시점의 은닉 상태 계산을 위한 입력값으로 사용한다.

![https://wikidocs.net/images/page/22886/rnn_image2_ver3.PNG](https://wikidocs.net/images/page/22886/rnn_image2_ver3.PNG)

- 피드 포워드 신경망에서는 뉴런이라는 단위를 사용했지만, RNN에서는 뉴런이라는 단위보다는 입력층과 출력층에서는 각각 입력 벡터와 출력 벡터, 은닉층에서는 은닉 상태 라는 표현을 주로 사용한다.

![https://wikidocs.net/images/page/22886/rnn_image2.5.PNG](https://wikidocs.net/images/page/22886/rnn_image2.5.PNG)

RNN을 굳이 뉴런 단위로 시각화 하면 이러한 모양

- RNN은 입력과 출력의 길이를 다르게 설계 할 수 있으므로, 다양한 용도로 사용할 수 있다.

![https://wikidocs.net/images/page/22886/rnn_image3_ver2.PNG](https://wikidocs.net/images/page/22886/rnn_image3_ver2.PNG)

> RNN에 대한 수식

![https://wikidocs.net/images/page/22886/rnn_image4_ver2.PNG](https://wikidocs.net/images/page/22886/rnn_image4_ver2.PNG)

- 현재 시점 t에서의 은닉 상태값을 $h_{t}$라고 정의한다면, 은닉층의 메모리 셀은 $h_{t}$를 계산하기 위해서 총 두 개의 가중치를 갖게 된다. 입력층에서 입력을 위한 가중치는 $W_{x}$이고, 하나는 이전 시점 t-1의 은닉 상태값인 $h_{t-1}$을 위한 가중치 $W_{h}$이다.

$$은닉층 = h_{t} = tanh(W_{x} x_{t} + W_{h}h_{t−1} + b)\\출력층 = y_{t} = f(W_{y}h_{t} + b)$$

- RNN의 은닉층 연산은 벡터와 행렬 연산으로 이해할 수 있다.

![https://wikidocs.net/images/page/22886/rnn_images4-5.PNG](https://wikidocs.net/images/page/22886/rnn_images4-5.PNG)

## 2. 케라스(Keras)로 RNN 구현하기

```python
modal.add(SimpleRNN(hidden_size, ...))

# 추가 인자를 사용할 때
model.add(SimpleRNN(hidden_size, input_shape=(timesteps, input_dim)))

# 다른 표기
model.add(SimpleRNN(hidden_size, input_length=M, input_dim=N))
# 단, M과 N은 정수
```

hidden_size: 은닉 상태의 크기를 정의, 중소형 모델의 경우 보통 128, 256, 512, 1024 등의 값을 가진다.

timesteps : 입력 시퀀스의 길이(Input_length)라고 표현하기도 함. 시점의 수

input_dim : 입력의 크기

![https://wikidocs.net/images/page/22886/rnn_image6between7.PNG](https://wikidocs.net/images/page/22886/rnn_image6between7.PNG)

- RNN 층은 batch_size, timesteps, tinput_dim 크기의 3D 텐서를 입력으로 받는다. batch_size는 한 번에 학습하는 데이터의 개수를 말한다.
- 위의 코드가 리턴하는 결과값은 출력층의 결과가 아니라, 하나의 은닉 상태 또는 정의하기에 따라 다수의 은닉 상태이다. 아래의 그림은 출력층을 포함한 완성된 인성 신경망그림과 은닉층까지만 표현하 그림의 차이를 보여준다.

![https://wikidocs.net/images/page/22886/rnn_image7_ver2.PNG](https://wikidocs.net/images/page/22886/rnn_image7_ver2.PNG)

- RNN 층은 사용자의 설정에 따라 2가지 종류의 출력을 내보낸다. 메모리 셀의 최종 시점의 은닉 상태만을 리턴하고자 한다면 (batch_size, output_dim) 크기의 2D 텐서를 리턴한다. 하지만 메모리 셀의 각 시점 (time step)의 은닉 상태값들을 모아서 전체 시퀀스를 리턴하고자 한다면 (batch_size, timesteps, output_dim) 크기의 3D 텐서를 리턴한다. 이는 RNN 층의 return_sequences 매개 변수에 True를 설정하여 설정이 가능하다.
- 마지막 은닉 상태만 전달하도록 하면, many-to-one 문제를 풀 수 있고, 모든 시점의 은닉 상태를 전달하도록 하면 다음층에 은닉층이 하나 더 있는 경우이거나 many-to-many 문제를 풀 수 있다.

```python
from keras.models import Sequential
from keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2,10)))
# model.add(SimpleRNN(3, input_length=2, input_dim=10))와 동일함.
model.summary()
```

- 위의 코드는 hidden_size의 값은 3, batch_size는 현 단계에서는 알 수 없으므로, (None, 3)이 된다.

## 3. 직접 RNN 구현하기

$$h_{t} = tanh(W_{x}X_{t} + W_{h}h_{t−1} + b)$$

```python
hidden_state_t = 0 # 초기 은닉 상태를 0(벡터)으로 초기화
for input_t in input_length: # 각 시점마다 입력을 받는다.
    output_t = tanh(input_t, hidden_state_t) # 각 시점에 대해서 입력과 은닉 상태를 가지고 연산
    hidden_state_t = output_t # 계산 결과는 현재 시점의 은닉 상태가 된다.
```

```python
import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_dim = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_dim)) # 입력에 해당되는 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.
# [0. 0. 0. 0. 0. 0. 0. 0.]
```

```python
Wx = np.random.random((hidden_size, input_dim))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치. (Dh * d)
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치. (Dh * Dn)
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias). (bh * 1)
```

```python
total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs: # 각 시점에 따라서 입력값이 입력됨.
	# inputs
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b) # Wx * Xt + Wh * Ht-1 + b(bias)
  total_hidden_states.append(list(output_t)) # 각 시점의 은닉 상태의 값을 계속해서 축적
  print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
  hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis = 0)
# 출력 시 값을 깔끔하게 해준다.

print(total_hidden_states) # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.

# 시점의 수 ( timesteps, t ) is 10
# 입력의 차원 input_dim is 4
# 은닉상태의 크기 hidden_size is 8
# 편향은 = hidden_size * 1 크기의 행렬
# 입력치에 대한 가중치는 = hidden_size * input_dim Wx
# 은닉상태에 대한 가중치는 = hidden_size * hidden_size Wh

# 때문에
# 각 차원별 입력치는 input_t
# np.dot(Wx, input_t) = 입력 가중치와 입력치에 대한 연산 1
# np.dot(Wh, hidden_state_t) = 이전 시점 값에 대한 연산 2
# + bias 편향 계산
# 마지막 output_dim은 당연하게도, output_dim hidden_state_size가 된다.
```

## 4. 깊은 순환 신경망 (Deep Recurrent Neural Network)

- RNN도 다수의 은닉층을 가질 수 있다고 언급한 적이 있다.

![https://wikidocs.net/images/page/22886/rnn_image4.5_finalPNG.PNG](https://wikidocs.net/images/page/22886/rnn_image4.5_finalPNG.PNG)

## 5. 양방향 순환 신경망 (Bidirectional Recurrent Neural Network)

- 양방향 순환 신경망은 시점 t에서의 출력값을 예측할 때 이전 시점의 데이터뿐만 아니라, 이후 데이터로도 예측할 수 있다는 아이디어에 기반한다.
- RNN은 과거 시점의 데이터들을 참고해서, 찾고자하는 정답을 예측하지만 실제 문제에서는 과거 시점의 데이터만 고려하는 것이 아니라, 향후 시점의 데이터에 힌트가 있는 경우도 많다. 그래서 이전 시점의 데이터뿐만 아니라, 이후 시점의 데이터도 힌트로 활용하기 위해서 고안된 것이 양방향 RNN이다.

![https://wikidocs.net/images/page/22886/rnn_image5_ver2.PNG](https://wikidocs.net/images/page/22886/rnn_image5_ver2.PNG)

- 양방향 RNN은 하나의 출력값을 예측하기 위해 기본적으로 2개의 메모리 셀을 사용한다. 첫번째 메모리 셀은 앞서 배운 앞 시점의 은닉상태(Forward States)를 전달받아 현재의 은닉 상태를 계산한다.
- 두번째 메모리 셀은 뒤 시점의 은닉 상태(Backward States)를 전달 받아 현재의 은닉 상태를 계산한다.
- 그리고 이 두개의 값 모두가 출력층에서 출력값을 예측하기 위해 사용된다.
- 양방향 RNN도 다수의 은닉층을 가질 수 있다.
- 다른 인공 신경망 모델들도 마찬가지이지만, 은닉층을 무조건 추가한다고 해서 모델의 성능이 좋아지는 것은 아니다. 은닉층을 추가하면, 학습할 수 있는 양이 많아지지만 또한 반대로 훈련 데이터 또한 그만큼 많이 필요하다.

# 2. 장단기 메모리 (Long Short-Term Memory, LSTM)

## 1. Vanila RNN의 한계

- RNN은 비교적 짧은 시퀀스에 대해서만 효과를 보이는 단점이 있다. 바닐라 RNN의 시점이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생한다.
- 뒤로 갈수록 앞에 입력되는 입력값의 정보량이 손실되어감으로서, 시점이 충분히 긴 상황에서는 앞에 입력 값들의 전체 정보에 대한 영향력은 거의 의미가 없다고 볼 수 있게 된다.
- 데이터는 가장 중요한 정보가 시점의 앞 쪽에 위치하게 될 수도 있다.

  → ''모스크바에 여행을 왔는데 건물도 예쁘고 먹을 것도 맛있었어. 그런데 글쎄 직장 상사한테 전화가 왔어. 어디냐고 묻더라구 그래서 나는 말했지. 저 여행왔는데요. 여기 \_\_\_''

  → 이때 RNN이 충분한 기억력을 가자고 있지 못한다면 다음 단어를 엉뚱하게 예측할 것이다.

  → 이를 장기 의존성 문제(the problem of Long-Term Dependencies) 라고 한다.

## 2. 바닐라 RNN 내부 열어보기

![https://wikidocs.net/images/page/22888/vanilla_rnn_ver2.PNG](https://wikidocs.net/images/page/22888/vanilla_rnn_ver2.PNG)

- 바닐라 RNN은 $x_{t}$와 $h_{t-1}$이라는 두 개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 입력이 된다. 그리고 이를 하이퍼볼릭탄젠트 함수의 입력으로 사용하고, 이 값은 은닉층의 출력인 은닉 상태가 된다.

## 3. LSTM(Long Short-Term Memory)

![https://wikidocs.net/images/page/22888/vaniila_rnn_and_different_lstm_ver2.PNG](https://wikidocs.net/images/page/22888/vaniila_rnn_and_different_lstm_ver2.PNG)

- LSTM은 은닉층의 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정의한다.
- 요약하면 LSTM은 은닉 상태를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며, 셀 상태(cell state)라는 값을 추가하였다. 위의 그림에서는 t시점의 셸 상태를 $C_{t}$로 표현하고 있다.
- LSTM은 RNN과 비교하여 긴 시퀀스의 입력을 처리하는데, 탁월한 성능을 보인다.
- 셸 상태 또한 이전에 배운 은닉 상태처럼 이전 시점의 셸 상태가 다음 시점의 셸 상태를 구하기 위한 입력으로 사용된다.
- 은닉 상태값과 셸 상태값을 구하기 위해서 새로 추가 된 3개의 게이트를 사용한다. 각 게이트는 삭제 게이트, 입력 게이트, 출력 게이트라고 부르며, 이 3개의 게이트에는 공통적으로 시그모이드 함수가 존재한다.
- 시그모이드 함수를 지나면 0과 1사이의 값이 나오게 되는데, 이 값들을 가지고 게이트를 조절한다.

> Reference

- σ는 시그모이드 함수를 의미한다.
- tanh는 하이퍼볼릭탄젠트 함수를 의미한다.
- $W_{xi}, W_{xg}, W_{xf}, W_{xo}$는 $x_{t}$와 함께 각 게이트에서 사용되는 4개의 가중치이다.
- $W_{hi}, W_{hg}, W_{hf}, W_{ho}$는 $h_{t-1}$와 함께 각 게이트에서 사용되는 4개의 가중치이다.
- $b_{i}, b_{g}, b_{f}, b_{o}$는 각 게이트에서 사용되는 4개의 편향이다.

> 입력 게이트

![https://wikidocs.net/images/page/22888/inputgate.PNG](https://wikidocs.net/images/page/22888/inputgate.PNG)

$$i_{t}=σ(W_{xi}x_{t}+W_{hi}h_{t-1}+b_{i})\\g_{t}=tanh(W_{xg}x_{t}+W_{hg}h_{t-1}+b_{g})$$

- 입력 게이트는 현재 정보를 기억하기 위한 게이트이다. 현재 시점 t의 x값과 입력 게이트로 이어지는 가중치 $W_{xi}$를 곱한값과 이전 시점의 t-1의 은닉 상태가 입력 게이트로 이어지는 가중치 $W_{xi}$를 곱한 값을 더하여 시그모이드 함수를 지난다. 이를 $i_{t}$라고 한다.
- 현재 시점 t의 x값과 입력 게이트로 이어지는 $W_{xi}$를 곱한 값과 이전 시점 t-1의 은닉 상태가 입력 게이트로 이어지는 가중치 $W_{hg}$를 곱한 값을 더하여 하이퍼 볼릭 탄젠트 함수를 지난다. 이를 $g_{t}$라고 한다.
- 시그모이드 함수를 지나 0과 1사이의 값과 하이퍼볼릭탄젠트 함수를 지나 -1과 1사이의 값 두개가 나오게 되는데, 이 두개의 값을 가지고 이번에 선택된 기억할 정보의 양을 정한다.

> 삭제 게이트

![https://wikidocs.net/images/page/22888/forgetgate.PNG](https://wikidocs.net/images/page/22888/forgetgate.PNG)

$$f_{t}=σ(W_{xf}x_{t}+W_{hf}h_{t-1}+b_{f})$$

- 삭제 게이트는 기억을 삭제하기 위한 게이트 이다.
- 현재 시점 t의 x값과 이전 시점 t-1의 은닉 상태가 시그모이드 함수를 지나게 된다. 시그모이드 함수를 지나면 0과 1사이의 값이 나오게 되는데, 이 값이 곧 삭제 과정을 거친 정보의 양이다.
- 0에 가까울수록 정보가 많이 삭제된 것이고, 1에 가까울수록 정보를 온전히 기억한 것이다.

> 셀 상태 (장기 상태)

![https://wikidocs.net/images/page/22888/cellstate2.PNG](https://wikidocs.net/images/page/22888/cellstate2.PNG)

$$C_{t}=f_{t}∘C_{t-1}+i_{t}∘g_{t}$$

- 셀 상태 $C_{t}$를 LSTM에서는 장기 상태라고 부르기도 한다. 이 시점에서는 삭제 게이트에서 일부 기억을 잃은 상태이다.
- 입력 게이트에서 구한 $i_{t}, g_{t}$이 두개의 값에 대해서 원소별 곱을 진행한다. 다시말해 같은 크기의 두 행렬이 있을 때 같은 위치의 성분끼리 곱한 것을 말한다. 여기서는 ∘ 로 표현되었다. 이것이 이번에 선택된 기억할 값이다.
- 입력 게이트에서 선택된 기억을 삭제 게이트의 결과값과 더한다. 이 값을 현재 시점의 t의 셸 상태라고 하며, 이 값은 다음 t+1 시점의 LSTM 셀로 넘겨진다.
- 삭제 게이트와 입력 게이트의 영향력을 이해해보자. 만약 삭제 게이트의 출력값이 0이 된다면, 이전 시점의 상태값은 현재 시점의 상태값을 결정하기 위한 영향력이 0이 되면서, 오직 입력 게이트의 결과만이 현재 시점의 셀 상태값을 결정할 수 있다.
- 이는 삭제 게이트가 완전히 닫히고, 입력 게이트를 연 상태를 의미한다.
- 반대로 입력 게이트의 값은 0이라고 한다면 오직 이전 시점의 셸 상태값에만 의존을 한다. 이는 입력 게이트를 완전히 닫고, 삭제 게이트만을 연 상태를 의미한다.
- 결과적으로 삭제 게이트는 이전 시점의 입력을 얼마나 반영할지를 의미하고, 입력 게이트는 현재 시점의 입력을 얼마나 반영할지를 결정한다.

> 출력 게이트와 은닉 상태(단기 상태)

![https://wikidocs.net/images/page/22888/outputgateandhiddenstate.PNG](https://wikidocs.net/images/page/22888/outputgateandhiddenstate.PNG)

$$o_{t}=σ(W_{xo}x_{t}+W_{ho}h_{t-1}+b_{o})\\h_{t}=o_{t}∘tanh(c_{t})$$

- 출력 게이트는 현재 시점 t의 x값과 이전 시점 t-1의 은닉 상태가 시그모이드 함수를 지난 값이다. 해당 값은 현재 시점 t의 은닉 상태를 결정하는 일에 쓰이게 된다.
- 은닉 상태를 단기 상태라고도 표현한다.
- 단기 상태의 값은 또한 출력층으로도 향한다.

# 3. 케라스의 SimpleRNN과 LSTM 이해하기

## 1. 임의의 입력 생성하기

```python
train_X = [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]
print(np.shape(train_X))
```

- 해당 입력은 단어 벡터의 차원은 5이고, 문장의 길이가 4인 경우를 가정한 입력이다.
- 다시 말해 4번의 시점(timesteps)이 존재하고, 각 시점마다 5차원의 단어 벡터가 입력으로 사용된다.
- **하지만 RNN은 2D 텐서가 아니라, 3D 텐서를 입력 받는다**고 언급했다. 즉, 위에서 만든 2D 텐서를 3D 텐서로 변경한다.
- 이는 배치 크기 1을 추가해주므로서 해결한다.

```python
train_X = [[[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]]
train_X = np.array(train_X, dtype=np.float32)
print(train_X.shape)
```

- batch_size는 한번에 RNN이 학습하는 데이터의 양을 의미하지만, 여기서는 샘플이 1개 밖에 없으므로 batch_size는 1이다.

## 2. SimpleRNN 이해하기

- SimpleRNN에는 여러 인자가 있으며, 대표적인 인자로 return_sequences와 return_state가 있다. 둘 다 False로 지정되어져 있으므로, 별도 지정을 하지 않을 경우에는 False로 처리된다.
- 실습에서 SimpleRNN을 매번 재선언하므로, 은닉 상태의 값 자체는 매번 초기화되어 이전 출력과 값의 일관성은 없고, 출력값보다는 해당 값의 크기에 주목해야 한다. (shape)

```python
rnn = SimpleRNN(3)
# rnn = SimpleRNN(3, return_sequences=False, return_state=False)와 동일.
hidden_state = rnn(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))

# hidden state : [[-0.34527388  0.958614    0.9838553 ]], shape: (1, 3)
```

```python
rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))

# hidden states : [[[ 0.99491495  0.9931372   0.99746007]
#  [ 0.9368357   0.9998274   0.9997208 ]
#  [-0.3199736   0.999477    0.9998163 ]
#  [ 0.7977573   0.9595988   0.77706915]]], shape: (1, 4, 3)
```

- (1,3) (1,4,3) 크기의 텐서가 출력된다.
- return_sequences 매개변수에 따른 결과값인데, 이는 시점에 따른 텐서 크기 출력(Ture), 마지막 시점만을 출력(False)에 관계한다.

## 3. LSTM 이해하기

- 실제로 SimpleRNN을 사용하는 경우는 거의 없다. 이보다는 LSTM이나 GRU를 주로 사용한다.

```python
lstm = LSTM(3, return_sequences=False, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))
```

- 이번에는 SimpleRNN 때와는 달리, 3개의 결과를 반환한다.
- LSTM과 SimpleRNN의 차이점은 return_state를 True로 둔 경우에는 마지막 시점의 은닉 상태뿐만 아니라, 셀 상태까지 반환한다는 점이다.

## 4. Bidirectional(LSTM) 이해하기

- 은닉 상태의 값을 고정 시킨 상태로 양방향 LSTM의 출력값을 확인해보자.

```python
k_init = tf.keras.initializers.Constant(value=0.1)
b_init = tf.keras.initializers.Constant(value=0)
r_init = tf.keras.initializers.Constant(value=0.1)
```

```python
bilstm = Bidirectional(LSTM(3, return_sequences=False, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))
```

- 무려 5개의 값을 반환한다.
- return_state가 True인 경우에는 정방향 LSTM의 은닉상태와 셸 상태, 역방향 LSTM의 은닉 상태와 셸 상태 4가지를 반환하기 때문이다.
- 첫 번째 출력값의 크기가 (1,6)인 것에 주목하자. 이는 return_sequences가 False인 경우, 정방향 LSTM의 마지막 시점의 은닉 상태와 역방향 LSTM의 첫번째 시점의 은닉 상태가 연결된 채 반환되기 때문이다.

```python
bilstm = Bidirectional(LSTM(3, return_sequences=True, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)
print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))
```

- 다음과 같이 return_sequences를 True로 둘 경우에는 hidden state의 출력값에서는 모든 시점의 은닉상태가 출력된다.
- 그 동시에 역방향 LSTM의 첫번째 시점의 은닉 상태는 더 이상 정방향 LSTM의 마지막 시점의 은닉 상태와 연결되는 것이 아니라 정방향 LSTM의 첫번째 시점의 은닉 상태와 연결된다.

# 5. RNN 언어 모델 (Recurrent Neural Network Language Model, RNNLM)

## 1. RNN 언어 모델 (Recurrent Neural Network Language Model, RNNLM)

- 앞서 n-gram 언어 모델과 NNLM은 고정된 개수의 단어만을 입력으로 받아야한다는 단점이 있었다.
- 하지만 시점(timestep)이라는 개념이 도입된 RNN으로 언어 모델을 만들면 입력의 길이를 고정하지 않을 수 있다.
- 이처럼 RNN으로 만든 언어 모델을 RNNLM(Recurrent Neural Network Language Model)이라고 한다.

> 예문 : what will the fat cat sit on

- 언어 모델은 주어진 단어 시퀀스로부터 다음 단어를 예측하는 모델이다.

![https://wikidocs.net/images/page/46496/rnnlm1_final_final.PNG](https://wikidocs.net/images/page/46496/rnnlm1_final_final.PNG)

- RNNLM은 기본적으로 예측 과정에서 이전 시점의 출력을 현재 시점의 입력으로 한다. RNNLM은 what을 입력받으면, will을 예측하고, 이 will은 다음 시점의 입력이 되어 the를 예측한다. 그리고 the는 또 다시 다음 시점의 입력이 되고, 해당 시점에는 fat을 예측한다. ( 테스트 과정 동안(실제 사용할 때 ) == 훈련이 끝남 )
- 훈련 과정에서는 이전 시점의 예측 결과를 다음 시점의 입력으로 넣으면서 예측하는 것이 아니라, what will the fat cat sit on이라는 훈련 샘플이 있다면, what will the fat cat sit 시퀀스를 모델의 입력으로 넣으면, will the fat cat sit on를 예측하도록 훈련된다.
- will the fat cat sit on이 각 시점의 레이블이다.

  → 이러한 RNN 훈련 기법을 교사 강요(teacher forcing)라고 한다. 이는 테스트 과정에서 t 시점의 출력이 t+1 시점의 입력으로 사용되는 RNN 모델을 훈련시킬 때 사용하는 훈련 기법이다.

  → 이러한 기법을 사용하면 모델이 t 시점에서 예측한 값을 t+1 시점에 입력으로 사용하지 않고, t 시점의 레이블. 즉, 실제 알고 있는 정답을 t+1 시점의 입력으로 사용한다.

  → 물론, 훈련 과정에서도 이전 시점의 출력을 다음 시점의 입력으로 사용하면서 훈련 시킬 수도 있지만 이는 한 번 잘못 예측하면 뒤에서의 예측까지 영향을 미쳐 훈련 시간이 느려지게 되므로, 교사 강요를 사용하여 RNN을 좀 더 빠르고 효과적으로 훈련시킬 수 있다.

![https://wikidocs.net/images/page/46496/rnnlm2_final_final.PNG](https://wikidocs.net/images/page/46496/rnnlm2_final_final.PNG)

- 훈련 과정 동안 출력층에서 사용하는 활성화 함수는 소프트맥스 함수이다.
- 그리고 모델이 예측한 값과 실제 레이블과의 오차를 계산하기 위해서 손실 함수로 크로스 엔트로피 함수를 사용한다.

![https://wikidocs.net/images/page/46496/rnnlm3_final.PNG](https://wikidocs.net/images/page/46496/rnnlm3_final.PNG)

- RNNLM의 구조를 봐보자. RNNLM은 위의 그림과 같이 총 4개의 층(layer)으로 이루어진 인공신경망이다.
- RNNLM의 현 시점을 4로 가정한다면, 4번째 입력 단어인 fat의 원-핫 벡터가 입력된다.
- 해당 모델이 예측해야하는 정답에 해당되는 단어 cat의 원-핫 벡터는 출력층에서 모델이 예측한 값의 오차를 구하기 위해 사용될 예정이다.
- 그리고 이 오차로부터 손실 함수를 사용해 인공 신경망이 학습을 하게 된다.
- 현 시점의 입력 단어의 원-핫 벡터 Xt를 입력받은 RNNLM은 우선 임베딩층을 지난다. 여기는 NNLM 챕터에서 룩업 테이블을 수행하는 투사층이었다. 이는 RNNLM에서는 임베딩층이라는 표현을 사용하도록 하겠다.
- 단어 집합의 크기가 V일 때, 임베딩 벡터의 크기를 M으로 설정하면, 각 입력 단어들은 임베딩층에서 V x M 크기의 임베딩 행렬과 곱해진다. 여기서 V는 단어 집합의 크기를 의미한다.
- 만약 원-핫 벡터의 차원이 7이고, M이 5라면 임베딩 행렬은 7 x 5 행렬이 된다. 그리고 임베딩 행렬은 역전파 과정에서 다른 가중치들과 함께 학습된다.

$$임베딩층:e_{t} = lookup(x_{t})\\은닉층:h_{t} = tanh(W_{x} e_{t} + W_{h}h_{t−1} + b)\\출력층:\hat{y_{t}} = softmax(W_{y}h_{t} + b)$$

1. 임베딩 벡터는 인늑칭에서 이전 시점의 은닉 상태인 $h_{t-1}$과 함께 tanh 연산을 하여 현재 시점의 은닉 상태 $h_{t}$를 계산하게 된다.
2. 출력층에서는 활성화 함수로 소프트맥스(softmax) 함수를 사용하는데, V차원의 벡터는소프트맥스 함수를 지나면서 각 원소는 0과 1사이의 실수값을 가지며, 총 합은 1이 되는 상태로 바뀐다.
3. 이렇게 나온 벡터를 RNNLM의 t시점의 예측값이라는 의미에서 $\hat{y}_{t}$라고 한다.
4. 벡터 $\hat{y}_{t}$의 각 차원 안에서의 값이 의미하는 것은 $\hat{y}_{t}$의 j번째 인덱스가 가진 0과 1사이의 값은 j번째 단어가 다음 단어일 확률을 나타낸다. 그리고 $\hat{y}_{t}$는 실제값. 즉, 실제 정답에 해당되는 단어인 원-핫 벡터의 값에 가까워져야 한다.
5. 실제값에 해당되는 다음단어를 $y$라고 했을 때, 이 두 벡터가 가까워지게 하기위해서 RNNLM는 손실 함수로 cross-entropy 함수를 사용한다.
6. 그리고 역전파가 이루어지면서 가중치 행렬들이 학습되는데, 이 과정에서 임베딩 벡터값들도 학습이 된다.
7. 룩업 테이블의 대상이 되는 테이블인 임베딩 행렬을 $E$라고 했을 때, 결과적으로 RNNLM에서 학습 과정에서 학습되는 가중치 행렬은 다음의 $E, W_x,W_h,W_y$4개가 된다.
