# 딥러닝 (Deep Learning)

- 딥 러닝의 기본 구조인 인공 신경망(Aritificial Neural Network)의 역사는 생각보다 오래되었다. 딥 러닝을 보다 손쉽게 이해하기 위해서 1957년에 등장한 초기 인공 신경망부터 학습해보도록 한다.
- 초기 신경망인 퍼셉트론(Perceptron)부터 피드 포워드 신경망 언어 모델의 정의, 그리고 기본적인 케라스의 사용법에 대해서 배워보자.

# 1. 퍼셉트론 (Perceptron)

- 인공 신경망은 수많은 머신 러닝 방법 중 하나이다. 하지만 최근 인공 신경망을 복잡하게 쌓아 올린 딥 러닝이 다른 머신 러닝 방법들을 뛰어넘는 성능을 보여주는 사례가 늘면서, 전통적인 머신 러닝과 딥 러닝을 구분해서 이해해야 한다는 목소리가 커지고 있다.

> **초기 인공 신경망 퍼셉트론**

## 1. 퍼셉트론 (Perceptron)

- 퍼셉트론(Perceptron)은 프랑크 로젠블란트가 1957년에 제안한 초기 형태의 인공 신경망이다.
- 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘이다.
- x의 입력값을 받고, 가중치(weight)를 통합해 y를 출력한다.
- 실제 신경 세포 뉴런에서의 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신한다. 각각의 인공 뉴런에서 보내진 입력값 x는 각각의 가중치 W와 함께 종착지인 인공 뉴런에 전달이 된다.
- **각 입력값이 가중치와 곱해져서 인공 뉴런에 보내**지고, **각 입력값과 그에 해당하는 가중치의 곱의 전체 합이 임계치(threshold)를 넘으면** **종착지에 있는 인공 뉴런은 출력 신호로서 1을 출력**하고, **그렇지 않을 경우에는 0을 출력**한다.
- 이러한 함수를 **계단 함수 (step function)**라고 한다.
- 계단 함수에 사용된 임계치값을 수식으로 표현할 때는 보통 세타(Θ)로 표현한다.

$$if \sum_i^{n} W_{i}x_{i}\ ≥ \theta → y=1 \\if \sum_i^{n} W_{i}x_{i}\ < \theta → y=0$$

- 해당 식은 임계치를 좌변으로 넘기고 편향(bias)로 표현할 수도 있다. 편향 또한 퍼셉트론의 입력으로 사용된다. 보통 그림에서는 1로 고정된다.

$$if \sum_i^{n} W_{i}x_{i} + b ≥ 0 → y=1 \\if \sum_i^{n} W_{i}x_{i} + b < 0 → y=0$$

- **뉴런에서 출력값을 변경시키는 함수를 활성화 함수(Activation Function)라고 한다.**
- 초기 인공 신경망 모델인 퍼셉트론은 활성화 함수로 계단 함수를 사용했지만, 계단 함수 외에도 여러 다양한 활성화 함수를 사용하기 시작했다.
- 시그모이드 함수나 소프트맥스 함수 또한 활성화 함수 중 하나이다.
- 로지스틱 회귀 모델이 인공 신경망에서는 하나의 인공 뉴런으로 볼 수 있다. 로지스틱 회귀를 수행하는 인공 뉴런과 퍼셉트론의 차이는 오직 활성화 함수의 차이이다.

## 2. 단층 퍼셉트론 (Single-Layer Perceptron)

- 입력값들, 가중치들, 편향, 하나의 출력의 한 세트를 단층 퍼셉트론이라고 한다.
- 이 때, 각 단계들을 층(layer)라고 부르는데, 입력값들의 구역을 입력층(input layer), 출력값의 구역을 출력층(output layer)이라고 한다.
- 단층 퍼셉트론을 이용하면 AND, NAND, OR 게이트를 쉽게 구현할 수 있다.
- 단층 퍼셉트론으로 구현이 불가능한 게이트가 존재한다. 바로 XOR게이트이다. XOR게이트는 입력값 2개가 서로 다른 값을 갖고 있을때에만 출력값이 1이 되고, 입력값 2개가 서로 같은 값을 가지면 출력값이 0이되는 게이트이다.
- 단층 퍼셉트론은 AND, NAND, OR 게이트와 같이 직선 하나로 두 영역을 나눌 수 있는 문제에 대해서만 구현이 가능하다.

![https://wikidocs.net/images/page/24958/andgraphgate.PNG](https://wikidocs.net/images/page/24958/andgraphgate.PNG)

- 하지만 XOR 게이트는 하얀색 원과 검은색 원을 직선 하나로 나누는 것이 불가능하다. 즉, 단층 퍼셉트론으로 XOR 게이트를 구현하는 것이 불가능한데, 단층 퍼셉트론은 선형 영역에 대해서만 분리가 가능하다는 한계점을 보여준다.
- XOR 게이트는 직선이 아닌, 곡선 비선형 영역으로 분리하면 구현이 가능하다.

![https://wikidocs.net/images/page/24958/xorgate_nonlinearity.PNG](https://wikidocs.net/images/page/24958/xorgate_nonlinearity.PNG)

## 3. 다층 퍼셉트론 (MultiLayer Perceptron, MLP)

- XOR 게이트는 기존의 AND, NAND, OR 게이트를 조합하면 만들 수 있다.
- 퍼셉트론 관점으로, 층을 더 쌓으면 만들 수 있다.
- 다층 퍼셉트론은 단층 퍼셉트론의 입력층과 출력층 사이에 층을 더 추가하는 것을 의미한다.
- **이렇게 입력층과 출력층 사이에 존재하는 층을 은닉층(hidden layer)라고 한다.**
- **즉, 다층 퍼셉트론은 중간에 은닉층이 존재한다는 점이 단층 퍼셉트론과 다르다.**

![https://wikidocs.net/images/page/24958/perceptron_4image.jpg](https://wikidocs.net/images/page/24958/perceptron_4image.jpg)

```python
def XOR_gate(x1, x2):
    nand_res = NAND_gate(x1,x2)
    or_res = OR_gate(x1, x2)

    res = AND_gate(nand_res, or_res)

    return res
```

- 이렇듯 XOR 문제보다 더욱 복잡한 문제를 해결하기 위해서 다층 퍼셉트론은 중간에 수많은 은닉층을 추가할 수 있다.

![https://wikidocs.net/images/page/24958/%EC%9E%85%EC%9D%80%EC%B8%B5.PNG](https://wikidocs.net/images/page/24958/%EC%9E%85%EC%9D%80%EC%B8%B5.PNG)

- 이렇듯, 은닉층이 2개 이상인 신경망을 심층 신경망(Deep Neural Network, DNN)이라고 한다.
- 퍼셉트론에 쓰이는 가중치 또한 기계가 스스로 찾아내도록 자동화 시켜야 한다. 이게 머신 러닝에서 말하는 학습(training) 단계에 해당되고, 선형 회귀와 로지스틱 회귀에서 보았듯이 손실 함수(Loss function)와 옵티마이저(Optimizer)를 사용한다.
- 학습을 시키는 인공 신경망이 심층 신경망일 경우에는 이를 심층 신경망을 학습시킨다고 하여, 딥 러닝(Deep Learning)이라고 한다.

# 2. 인공 신경망 (Artificial Neural Network)

## 1. 피드 포워드 신경망 (Feed-Forward Neural Network, FFNN)

![https://wikidocs.net/images/page/24987/mlp_final.PNG](https://wikidocs.net/images/page/24987/mlp_final.PNG)

- 다층 퍼셉트론(MLP)과 같이 입력층에서 출력층 방향으로 연산이 전개되는 신경망을 말한다.

## 2. 전결합층 (Fully-connected layer, FC, Dense layer)

- 모든 뉴런이 이전 층의 모든 뉴런과 연결돼 있는 층을 전결합층이라고 한다.
- 이와 동일한 의미로 밀집층 (Dense layer)라고도 부른다.
- 전결합층만으로 구성된 피드 포워드 신경망이 있다면, 이를 전결합 피드 포워드 신경망이라고도 한다.

## 3. 활성화 함수 (Activation Function)

- 앞서 배운 퍼셉트론에서는 계단 함수를 통해 출력값이 0이 될지, 1이 될지를 결정한다고 했는데, 이러한 매커니즘은 실제 뇌를 구성하는 신경 세포 뉴런이 전위가 일정치 이상이 되면 시냅스가 서로 화학적으로 연결되는 모습을 모방한 것이다.
- 이렇게 은닉층과 출력층의 뉴런에서 출력값을 결정하는 함수를 활성화 함수라고 한다.

### 1. 활성화 함수의 특징 - 비선형 함수(Nonlinear function)

- 인공 신경망의 능력을 높이기 위해서는 은닉층을 계속해서 추가해야 하는데, 활성화 함수로 선형 함수를 사용하게 되면 은닉층을 쌓을 수가 없다.
- 선형 함수로는 은닉층을 여러번 추가하더라도 1회 추가한 것과 차이를 줄 수 없다.
- 물론 선형 함수를 사용한 층도 학습 가능한 가중치가 새로 생긴다는 점에서 분명히 의미는 있다. 이와 같이 선형 함수를 사용한 층을 활성화 함수를 사용하는 은닉층과 구분하기 위해서 선형층(linear layer)이나 투사층(projection layer)등의 다른 표현을 사용하여 표현하기도 한다.
- 비선형층 (nonlinear layer)

### 2. 계단 함수 (Step function)

![https://wikidocs.net/images/page/24987/step_function.PNG](https://wikidocs.net/images/page/24987/step_function.PNG)

### 3. 시그모이드 (Sigmoid function)와 기울기 손실

![https://wikidocs.net/images/page/60683/simple-neural-network.png](https://wikidocs.net/images/page/60683/simple-neural-network.png)

- 시그모이드 함수를 사용한 위 인공 신경망의 학습과정은 입력에 대해서 순전파(forward propagation) 연산을 하고, 순전파 연산을 통해 나온 예측값과 실제값의 오차를 손실함수(loss function)을 통해 계산하고, 이 손실을 미분을 통해서 기울기(gradient)를 구하고, 이를 통해 역전파(back propagation)를 수행한다.
- 시그모이드 함수의 문제점은 미분을 해서 기울기(gradient)를 구할 때 발생한다.
- 시그모이드 함수는 0또는 1에 가까워지면 기울기가 완만해지는 모습을 보이는데, 거의 0에 가까운 수준으로 나타나게 된다.
- 그런데 역전파 과정에서 0에 가까운 아주 작은 기울기가 곱해지게 되면, 앞단에는 기울기가 잘 전달되지 않는다. 이러한 현상을 기울기 손실(Vanishing Gradient) 문제 라고 한다.
- 기울기의 전달이 제대로 되지 않으면 매개변수 W가 업데이트 되지 않아 학습이 되지를 않는다.
- 결론적으로 시그모이드 함수를 은닉층에서 사용하는 것은 지양된다.

### 4. 하이퍼볼릭탄젠트 함수 (Hyperbolic tangent function)

- 해당 함수는 입력값을 -1과 1사이의 값으로 변환한다.
- 모양 자체는 시그모이드 함수와 비슷하지만, 0을 중심으로 하기 때문에 반환값의 변화폭이 더 커져서 기울기 손실 증상이 적은편이다.

### 5. 렐루 함수 (ReLU)

- 렐루 함수는 음수를 입력하면 0을 출력하고, 양수를 입력하면 입력값을 그대로 반환한다. 이는 특정 양수값에 수렴하지 않으므로 시그모이드 함수에서 인공 신경망에서 훨씬 더 잘 작동한다.
- 하지만 입력값이 음수면 기울기가 0이된다는 단점이 존재한다. 이 뉴런을 다시 회생하는 것은 매우 어렵다. 이를 죽은 렐루(dying ReLU)라고 한다.

### 6. 리키 렐루 (Leaky ReLU)

- 죽은 렐루를 보완하기 위한 함수로, 입력값이 음수일 경우에는 0.001과 같은 매우작은 수를 변환하도록 설계되어있다.

### 7. 소프트맥스 함수 (Softmax function)

- 소프트맥스 함수는 시그모이드 함수처럼 출력층의 뉴런에서 주로 사용되는데, 시그모이드 함수가 2가지 선택지중 하나를 고르는 이진 분류에서 사용된다면, 소프트맥스 함수는 3가지 이상의 선택지 중 하나를 고르는 다중 클래스 분류 문제에 사용된다.

## 4. 행렬의 곱셈을 이용한 순전파 (Forward Propagation)

![https://wikidocs.net/images/page/24987/neuralnetwork_final.PNG](https://wikidocs.net/images/page/24987/neuralnetwork_final.PNG)

- 위 구성의 인공신경망을 만들고 싶다면 아래와 같이 keras에서 모델 정의를 한다.

```python
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() # 층을 추가할 준비
model.add(Dense(8, input_dim=4, init='uniform', activation='relu'))
# 입력층(4)과 다음 은닉층(8) 그리고 은닉층의 활성화 함수는 relu
model.add(Dense(8, activation='relu')) # 은닉층(8)의 활성화 함수는 relu
model.add(Dense(3, activation='softmax')) # 출력층(3)의 활성화 함수는 softmax
```

- 이렇게 인공 신경망에서 입력층에서 출력층 방향으로 연산을 진행하는 과정을 순전파라고 한다.
- 앞서 머신 러닝 챕터에서 배웠던 벡터와 행렬 연산을 인공 신경망에 적용하려고 하면, 벡터와 행렬 연산이 순전파 과정에서 층(layer)마다 적용된다.
- 순전파를 진행하고 예측값을 구하고나서 이 다음에 인공 신경망이 해야할 일은 예측값과 실제값으로부터 오차를 계산하고, 오차로부터 가중치와 편향을 업데이트하는 일이다. 즉, 인공 신경망의 학습 단계에 해당된다. 이때 인공 신경망은 순전파와는 반대 방향으로 연산을 진행하며 가중치를 업데이트하는데, 이 과정을 역전파(BackPropagation)라고 한다.

# 3. 딥 러닝의 학습 방법

## 1. 순전파 (Forward Propagation)

- 활성화 함수, 은닉층의 수, 각 은닉층의 뉴런 수 등 딥 러닝 모델을 설계하고나면 입력값은 입력층, 은닉층을 지나면서 각 층에서의 가중치와 함께 연산되며 출력층으로 향한다.
- 그리고 출력층에서 모든 연산을 마친 예측값이 나오게 된다.
- 이와 같이 입력층에서 출력층 방향으로 예측값의 연산이 진행되는 과정을 순전파라고 한다.

## 2. 손실 함수 (Loss Function)

- 손실 함수는 실제값과 예측값의 차이를 수치화해주는 함수이다.
- 이 두 값의 차이, 즉. 오차가 클수록 손실 함수의 값은 크고 오차가 작을 수록 손실 함수의 값은 작아진다.
- 회귀에서는 평균 제곱 오차, 분류 문제에서는 크로스 엔트로피를 주로 손실 함수로 사용한다.
- 손실 함수의 값을 최소화 하는 두 개의 매개변수인 가중치 W와 b를 찾아가는 것이 딥 러닝의 학습 과정이므로, 손실 함수의 선정은 매우 중요하다.

### 1. MSE(Mean Squared Error, MSE)

![https://wikidocs.net/images/page/24987/mse.PNG](https://wikidocs.net/images/page/24987/mse.PNG)

- 오차 제곱 평균으로, 연속형 변수를 예측할 때 사용된다.

### 2. 크로스 엔트로피(Cross-Entropy)

![https://wikidocs.net/images/page/24987/%ED%81%AC%EB%A1%9C%EC%8A%A4%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.PNG](https://wikidocs.net/images/page/24987/%ED%81%AC%EB%A1%9C%EC%8A%A4%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.PNG)

- 낮은 확률로 예측해서 맞추거나, 노은 확률로 예측해서 틀리는 경우 loss가 더 크다.
- keras의 model.compile()에서는 이진 분류의 경우 binary_crossentropy를 사용한다.
- 다중 클래스 분류일 경우 categorical_crossentropy를 사용한다.

## 3. 옵티마이저 (Optimizer)

- 손실 함수의 값을 줄여나가면서 학습하는 방법은 어떤 옵티마이저를 사용하느냐에 따라 달라진다.
- 여기서 배치(Batch)라는 개념에 대한 이해가 필요하다.
- **배치는 가중치 등의 매개변수의 값을 조정하기 위해 사용하는 데이터의 양을 말한다.**

### 1. 배치 경사 하강법 (Batch Gradient Descent)

- 배치 경사 하강법은 옵티마이저 중 하나로 오차(Loss)를 구할 때 전체 데이터를 고려한다.
- 머신 러닝에서는 1번의 훈련 횟수를 1 에포크라고 하는데, 배치 경사 하강법은 한 번의 에포크에 모든 매개변수 업데이트를 단 한 번 수행한다.
- 배치 경사 하강법은 전체데이터를 고려해서 학습하므로, 에포크 당 시간이 오래 걸리며, 메모리를 크게 요구한다는 단점이 있으나, 글로벌 미니멈을 찾을 수 있다는 장점이 있다.

### 2. 확률적 경사 하강법 (Stochastic Gradient Descent, SGD)

- 확률적 경사 하강법은 매개변수 값을 조정 시 전체 데이터가 아니라, 랜덤으로 선택한 하나의 데이터에 대해서만 계산하는 방법이다. 더 적은 데이터를 사용하므로, 더 빠르게 계산할 수 있다.
- 속도만큼은 배치 경사 하강법보다 빠르다는 장점이 있다.

### 3. 미니 배치 경사 하강법 (Mini-Batch Gradient Descent)

- 정해진 양에 대해서만 계산하여 매개 변수의 값을 조정하는 경사 하강법
- SGD 보다는 안정적이고, 일반 배치 경사 하강법보다는 빠르다는 장점이 있다.

### 4. 모멘텀 (Momentum)

- 관성이라는 물리학의 법칙을 응용한 방법이다.
- 경사 하강법에 관성을 더해준 것인데, 로컬 미니멈에 도달하였을 때, 기울기가 0이라서 기존의 경사 하강법이라면 글로벌 미니멈으로 잘못 인식하여 계산이 끝났을 상황이라도 모멘텀,. 즉, 관성의 힘을 빌리면 값이 조절되면서 로컬 미니멈에서 탈출하는 효과를 얻을 수도 있다.

### 5. 아다그라드 (Adagrad)

- 각 매개변수에 다른 학습률을 적용시킨다.
- 이 때, 변화가 많은 매개변수는 학습률이 작게 설정되고, 변화가 적은 매개변수는 학습률을 높게 설정한다.

### 6. 알엠에스프롭(RMSprop)

- 아다그라드 수식 대체

### 7. 아담(Adam)

- 알엠에스프롭에 모멘텀 두 가지를 합친 듯한 방법으로, 방향과 학습률 두 가지를 모두 잡기 위한 방법이다.

## 4. 에포크와 배치 크기와 이터레이션

### 1. 에포크(Epoch)

- 에포크란 인공 신경망에서 전체 데이터에 대해서 순전파와 역전파가 끝난 상태를 말한다.
- 에포크 횟수가 지나치거나 너무 적으면 과적합과 과소적합이 발생할 수 있다.

### 2. 배치 크기(Batch size)

- 몇 개의 데이터 단위로 매개변수를 업데이트 하는지를 말한다.
- 배치 크기는 배치의 수와는 다른 개념이다. 전체 데이터가 2000개일때 배치 크기를 200으로 준다면 배치의 수는 10이다. 이때 배치의 수를 이터레이션이라고 한다.

### 3. 이터레이션(Iteration)

- 한 번의 에포크를 끝내기 위해서 필요한 배치의 수를 말한다.
- SGD를 이 개념들을 가지고 설명하면, 배치 크기가 1이므로 모든 이터레이션마다 한의 데이터를 선택하여 경사 하강법을 수행한다.

## 5. 역전파 (BackPropagation)

### 1. 인공 신경망의 이해

### 2. 순전파 (Forward Propagation)

- 과정은 jupyter notebook을 통해 이해하도록 하자.
- 출력층까지의 값을 모두 더했으면, 이제 오차를 구하면 된다.
- 오차를 계산하기 위한 손실함수로는 MSE를 사용한다.

$$o_{1}=sigmoid(z_{3})=0.60944600\\o_{2}=sigmoid(z_{4})=0.66384491$$

- 다음과 같을 때 오차는 아래와 같이 구하게 된다.

$$E_{o1}=\frac{1}{2}(target_{o1}-output_{o1})^{2}=0.02193381\\E_{o2}=\frac{1}{2}(target_{o2}-output_{o2})^{2}=0.00203809\\E_{total}=E_{o1}+E_{o2}=0.02397190$$

### 3. 역전파 1단계 (BackPropagation Step 1)

- c출력층 바로 이전의 은닉층을 N층이라고 했을 때, 출력층과 N층 사이의 가중치를 업데이트하는 단계를 역전파 1단계라고 한다.
- 가중치를 업데이트 하기 위해서는 $\frac{∂E_{total}}{∂W_{i}}$을 진행해야 한다. 이는 다음과 같이 계산하면 된다.

$$\frac{∂E_{total}}{∂W_{5}} = \frac{∂E_{total}}{∂o_{1}} \text{×} \frac{∂o_{1}}{∂z_{3}} \text{×} \frac{∂z_{3}}{∂W_{i}}$$

- 예제를 통해 확인하도록 하자.

### 4. 역전파 2단계 (BackPropagation Step 2)

- 위와 같은 방법으로 똑같이 진행하면 된다.
- 인공 신경망의 학습은 오차를 최소화하는 가중치를 찾는 목적으로 순전파와 역전파를 반복하는 것을 말한다.

# 4. 과적합 (overfitting)을 막는 방법들

## 1. 데이터의 양을 늘리기

- 데이터의 양이 적으면 해당 데이터의 특정 패턴이나 노이즈까지 쉽게 암기하므로, 과적합 현상이 발생할 확률이 늘어난다. 그렇기 때문에 데이터의 양을 늘릴 수록 모델은 데이터의 일반적인 패턴을 학습하여 과적합을 방지할 수 있다.
- 데이터의 양이 적으면 의도적으로 기존의 데이터를 조금씩 변형하고, 추가하여 데이터의 양을 늘리기도 하는데 이를 데이터 증식 또는 증강(Data Augmentation)이라고 한다.

## 2. 모델의 복잡도 줄이기

- 인공 신경망의 복잡도를 줄인다.
- 인공 신경망에서는 모델에 있는 매개변수들의 수를 모델의 수용력(capacity)라고 하기도 한다.

## 3. 가중치 규제(Regularization) 적용하기

- 복잡한 모델이 간단한 모델보다 과적합될 가능성이 높다.

  → 여기서 말하는 간단한 모델은 적은 수의 매개변수를 가진 모델을 말한다.

- 복잡한 모델을 좀 더 간단하게 하는 방법으로 가중치 규제가 있다.

  → L1 규제 : 가중치 w들의 절대값 합계를 비용함수에 추가한다.

  → L2 규제 : 모든 가중치 w들의 제곱합을 비용 함수에 추가한다.

## 4. 드롭아웃 (dropout)

- 드롭아웃은 학습 과정에서 신경망의 일부를 사용하지 않는 방법이다.

# 5. 기울기 소실 (Gradient Vanishing)과 폭주(Exploding)

- 깊은 긴공 신경망을 학습하다보면 역전파 과정에서 입력층으로 갈 수록 기울기(Gradient)가 점차적으로 작아지는 현상이 발생할 수 있다.
- 입력층에 가까운 층들에서 가중치들이 업데이트가 제대로 되지 않으면 결국 최적의 모델을 찾을 수 없게 된다. 이를 기울기 소실(Gradient Vanishing)이라고 한다.
- 또한 반대로 기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산되기도 한다. 이를 기울기 폭주(Gradient Exploding)이라고 한다. 주로 순환 신경망에서 발생할 수 있다.

## 1. ReLU와 ReLU의 변형들

- 기울기 소실을 완화하는 가장 간단한 방법은 은닉층의 활성화 함수로 시그모이드나 하이퍼볼릭탄젠트 함수 대신에 ReLU의 변형 함수와 같은 Leaky ReLU를 사용하느 ㄴ것이다.

## 2. 그래디언트 클리핑 (Gradient Clipping)

- 그래디언트 클리핑은 기울기 값을 자르는 것을 의미한다. 기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 자른다. 다시 말해서 임계치만큼 크기를 감소시킨다. 이는 RNN에서 유용하다.

## 3. 가중치 초기화 (Weight initialization)

- 같은 모델을 훈련시키더라도 가중치가 초기에 어떤 값을 가졌느냐에 따라서 모델의 훈련 결과가 달라지기도 한다.

### 1. 세이비어 초기화(Xavier Initialization)

- 이 방법은 균등 분포 또는 정규 분포로 초기화할 때 두 가지 경우로 나뉘며, 이전 층의 뉴런 개수와 다음 층의 뉴런 개수를 가지고 식을 세운다.
- 세이비어 초기화는 여러 층의 기울기 분산 사이에 균형을 맞춰서 특정 층이 너무 주목을 받거나 다른 층이 뒤쳐지는 것을 막는다.
- 해당 초기화는 시그모이드 함수나 하이퍼볼릭 탄젠트 함수와 같은 S자 형태인 활성화 함수와 함께 사용할 경우에는 좋은 성능을 보이지만, ReLU와 함께 사용할 경우에는 성능이 좋지 않다.

### 2. He 초기화 (He initialization)

- He 초기화(He initialization)는 세이비어 초기화와 유사하게 정규 분포와 균등 분포 두 가지 경우로 나뉜다. 다만, He 초기화는 세이비어 초기화와 다르게 다음 층의 뉴런 수를 반영하지 않는다.

## 4. 배치 정규화 (Batch Normalization)

- 배치 정규화는 인공 신경망의 각 층에 들어가는 입력을 평균과 분산으로 정규화하여 학습을 효율적으로 만든다.

### 1. 내부 공변량 변화 (Internal Covariate Shift)

- 내부 공변량 변화란 학습 과정에서 층 별로 입력 데이터 분포가 달라지는 현상을 말한다.
- 이전 층들의 학습에 의해 이전 층의 가중치 값이 바뀌게 되면, 현재 층에 전달되는 입력 데이터의 분포가 현재 층이 학습했던 시점의 분포와 차이가 발생한다.
- 공변량 변화는 훈련 데이터의 분포와 테스트 데이터의 분포가 다른 경우를 의미한다.
- 내부 공변량 변화는 신경망 층 사이에서 발생하는 입력 데이터의 분포 변화를 의미한다.

### 2. 배치 정규화 (Batch Normalization)

- 배치 정규화는 표현 그대로 한 번에 들어오는 배치 단위로 정규화하는 것을 말한다.
- 배치 정규화는 각 층에서 활성화 함수를 통과하기 전에 수행된다.
  1. 입력에 대한 평균을 0으로 만들고, 정규화를 한다.
  2. 정규화 된 데이터에 대해서 스케일과 시프트를 수행한다.
  3. 매개변수 γ와 β를 사용하는데, γ는 스케일을 위해 사용하고, β는 시프트를 하는 것에 사용하며, 다음 레이어에 일정한 범위의 값들만 전달되게 한다.

### 3. 배치 정규화의 한계

> 미니 배치 크기에 의존적이다.

> RNN에 적용하기 어렵다.

- RNN은 각 시점(time step)마다 다른 통계치를 가진다. 이는 RNN에 배치 정규화를 적용하는 것을 어렵게 만든다.

## 5. 층 정규화 (Layer Normalization)

# 6. 케라스(keras) 훑어보기

[Keras: the Python deep learning API](https://keras.io/)

## 1. 전처리 (Preprocessing)

- Tokenizer() : 토큰화와 정수 인코딩을 위해 사용된다.
- pad_sequence() : 전체 훈련 데이터에서 각 샘플의 길이는 서로 다를 수 있다. 또는 각 문서 또는 각 문장은 단어의 수가 제각각이다. 모델의 입력으로 사용하려면 모든 샘플의 길이를 동일하게 맞추어야 한다.
  - 해당 작업을 패딩이라고 불렀따.
  - 보통 숫자 0을 넣어서 길이가 다른 샘플들의 길이를 맞추어 준다.

## 2. 워드 임베딩 (Word Embedding)

- 워드 임베딩이란 텍스트 내의 단어들을 밀집 벡터(dense vector)로 만드는 것을 말한다.
- 대부분의 값이 0인 이러한 벡터를 희소 벡터(sparse vector)라고 한다. (예: 원-핫 벡터)
- 원-핫 벡터는 단어의 수만큼 벡터의 차원을 가지며, 단어 간 유사도가 모두 동일하다는 단점이 있다.
- 반면, 희소 벡터와 표기상으로도 의미상으로도 반대인 벡터가 있다.
- 대부분의 값이 실수이고, 상대적으로 저차원인 밀집 벡터(dense vector)이다.

  → [0.1, -1.2, 0.8 ...] 과 같은 형식을 가진다.

- 단어를 이렇듯 밀집 벡터로 만드는 작업을 워드 임베딩이라고 한다.

  → 밀집 벡터는 워드 임베딩 과정을 통해 나온 결과이므로 임베딩 벡터 라고도 한다.

  → 원 핫 벡터의 차원이 주로 1만 단위로 큰것과는 달리 임베딩 벡터는 주로 256, 512, 1024 등의 차원을 가진다.

  → 임베딩 벡터는 초기에는 랜덤값을 가지지만, 인공 신경망의 가중치가 학습되는 방법과 같은 방식으로 값이 학습되며 변경된다.

- 케라스에서는 Embedding() 이라는 함수를 사용하여 단어를 밀집 벡터로 만든다. 이를 인공 신경망 용어로는 임베딩 층(embedding layer)을 만든다고 한다.

  → 해당 함수는 3D 텐서를 리턴한다.

## 3. 모델링 (Modeling)

- Sequential() : 케라스에서는 입력층, 은닉층, 출력층을 구성하기 위해 해당 메서드를 사용한다.

  - Sequential()을 model로 선언한 뒤에 model.add()라는 코드를 통해 층을 단계적으로 추가한다.

  ```python
  from tensorflow.keras.models import Sequential
  model = Sequential()
  model.add(...) # 층 추가
  model.add(...) # 층 추가
  model.add(...) # 층 추가
  ```

  - Embedding을 통해 생성하는 임베딩 층 또한 인공 신경망의 층의 하나이므로 model.add 해야 한다.

- Dense() : 전결합층(fully-connected layer)을 추가한다. model.add()를 통해 추가할 수 있다.

  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  model = Sequential()
  model.add(Dense(1, input_dim=3, activation='relu'))
  ```

  → 3개의 인자를 받는데, 출력 뉴런의 수, input_dim (입력 뉴런의 수), activation(활성화 함수)를 받는다.

  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  model = Sequential()
  model.add(Dense(8, input_dim=4, activation='relu'))
  model.add(Dense(1, activation='sigmoid')) # 출력층
  ```

  → 위와 같이 사용하면 아래와 같은 구성을 가지게 된다.

  ![https://wikidocs.net/images/page/32105/neural_network2_final.PNG](https://wikidocs.net/images/page/32105/neural_network2_final.PNG)

- summary() : 모델의 정보를 요약해서 보여준다.

## 4. 컴파일(Compile)과 훈련(Training)

- compile() : 모델을 기계가 이해할 수 있도록 컴파일 한다. 오차함수와 최적화 방법, 메트릭 함수를 선택할 수 있다.

```python
# 이 코드는 뒤의 텍스트 분류 챕터의 스팸 메일 분류하기 실습 코드를 갖고온 것임.
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
max_features = 10000

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32)) #RNN에 대한 설명은 뒤의 챕터에서 합니다.
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

- 위 코드는 임베딩층, 은닉층, 출력층을 추가하여 모델을 설계한 것 이고, compilie 과정에서 optimizer와 손실함수, 그리고 훈련을 모니터링 하기 위한 지표를 설정한 것이다.
- 케라스의 대표적인 손실 함수와 활성화 조합을 아래와 같다.
  - 회귀 문제 : mean_squared_error (평균 제곱 오차)
  - 다중 클래스 분류 : categorical_crossentropy (범주형 교차 엔트로피)
    - 소프트맥스
  - 다중 클래스 분류 : sparse_categorical_crossentrop
    - 소프트맥스 : 범주형 교차 엔트로피와 동일하지만 이 경우 원-핫 인코딩이 된 상태일 필요없이 정수 인코딩 된 상태에서 수행 가능
  - 이진 분류 : binary_crossentropy (이항 교차 엔트로피)
    - 시그모이드
- fit() : 모델을 학습한다. 모델이 오차로부터 매개 변수를 업데이트 시키는 과정을 학습, 훈련, 또는 적합(fitting)이라고 하기도 하는데, 모델이 데이터에 적합해가는 과정이기 때문이다.

```python
# 위의 compile() 코드의 연장선상인 코드
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

- 첫번째 인자 - 훈련 데이터
- 두번째 인자 - 레이블 데이터
- epochs - 총 훈련 횟수
- batch_size - 배치 크기, 기본값은 32, 미니 배치 경사 하강법을 사용하고 싶지 않을 경우에는 None 변수를 보내준다.
- validation_data(x_val, y_val) - 검증 데이터를 사용한다. 검증 데이터를 사용하면 각 에포크마다 검증 데이터의 정확도도 함께 출력되는데, 이 정확도는 훈련이 잘 되고 있는지를 보여줄 뿐이며, 이를 학습하지는 않는다.
- validation_split - validation_data 대신 사용할 수 있다.

  → 이 또한 훈련 자체에는 반영되지 않고, 훈련 과정을 지켜보기 위한 용도로 사용된다.

  → 훈련용 데이터에서 일정 비율을 때서 사용한다.

- verbose - 학습 중 출력되는 문구를 설정한다.

  → 0 : 아무것도 출력하지 않음

  → 1 : 훈련의 진행도를 보여주는 진행 막대를 보여준다.

  → 2 : 미니 배치마다 손실 정보를 출력한다.

## 5. 평가(Evaluation)와 예측(Prediction)

- evaluate() : 테스트 데이터를 통해 학습한 모델에 대한 정확도를 평가한다.
- predict() : 임의의 입력에 대한 모델의 출력값을 확인한다.

## 6. 모델의 저장(Save)과 로드(Load)

- save("파일명")
- load_model("파일명")

# 7. 케라스의 함수형 API

- 위에서 배운 keras 설명은 Sequential API를 사용한 모델 설계 방식 대로 설명을 한 것 이다. 하지만 Sequential API는 여러 층을 공유하거나 다양한 종류의 입력과 출력을 사용하는 등의 복잡한 모델을 만드는 일에는 한계가 존재한다.
- 더욱 복잡한 모델을 생성할 수 있는 방식으로는 Functional API가 있다.

## 1. Sequential API로 만든 모델

- 해당 모델은 직관적이고 편리하지만 단순히 층을 쌓는 것만으로는 구현할 수 없는 복잡한 신경망을 구현할 수 없다.

## 2. Functional API로 만든 모델

- Functional API는 각 층을 일종의 함수(function)로서 정의를 한다. 그리고 각 함수를 조합하기 위한 연산자들을 제공한다.

### 1. 전결합 피드 포워드 신경망 (Fully-connected FFNN)

- functional API에서는 입력 데이터의 크기를 인자로 입력층을 정의해주어야 한다.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(10,))
x = Dense(8, activation="relu")(inputs)
x = Dense(4, activation="relu")(x)
x = Dense(1, activation="linear")(x)
model = Model(inputs, x)
```

- 이와 같이 Input() 함수로 입력의 크기를 정의한다.
- 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당한다.
- Model() 함수에 입력과 출력을 정의한다.
- 이를 model로 저장하면 model.compile, [model.fit](http://model.fit) 등을 사용할 수 있다.

### 2. 선형 회귀 (Linear Regression)

- jupyter notebook 확인

### 3. 로지스틱 회귀 (Logistic Regression)

- jupyter notebook 확인

### 4. 다중 입력을 받는 모델 (model that accepts multiple inputs)

- functional API를 사용하면 다중 입력과 다중 출력을 가지는 모델도 만들 수 있다.

```python
# 최종 완성된 다중 입력, 다중 출력 모델의 예
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
```

```python
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# 두 개의 입력층을 정의
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

# 첫번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
x = Dense(16, activation="relu")(inputA)
x = Dense(8, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# 두번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(8, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# 두개의 인공 신경망의 출력을 연결(concatenate)
result = concatenate([x.output, y.output])

# 연결된 값을 입력으로 받는 밀집층을 추가(Dense layer)
z = Dense(2, activation="relu")(result)
# 선형 회귀를 위해 activation=linear를 설정
z = Dense(1, activation="linear")(z)

# 결과적으로 이 모델은 두 개의 입력층으로부터 분기되어 진행된 후 마지막에는 하나의 출력을 예측하는 모델이 됨.
model = Model(inputs=[x.input, y.input], outputs=z)
```

- 해당 예제는 concatenate라는 함수로 2개의 신경망의 출력을 연결하는 것이 핵심이다.

### 5. RNN (Recurrence Neural Network) 은닉층 사용하기

- LSTM 생성자는 numpy version 1.19 이전부터 작동한다.

```python
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
inputs = Input(shape=(50,1))
lstm_layer = LSTM(10)(inputs) # RNN의 일종인 LSTM을 사용
x = Dense(10, activation='relu')(lstm_layer)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)
```

# 8. 케라스 서브클래싱 API (Keras Subclassing API)

- 또 다른 모델 구현 방식.

![https://wikidocs.net/images/page/106897/1_WzwKtnA0LEhiCGdWTTpLaA.png](https://wikidocs.net/images/page/106897/1_WzwKtnA0LEhiCGdWTTpLaA.png)

- Subclassing API로는 functional API가 구현할 수 없는 모델들조차 구현할 수 있는 경우가 있다.
- Functional API는 기본적으로 딥 러닝 모델을 DAG(directed acyclic graph)로 취급한다. 실제로 대부분의 딥 러닝 모델이 이에 속하기는 하지만, 항상 그렇지는 않다.

  → 예를 들어서 재귀 네트워크나 트리 RNN은 이 가정을 따르지 않으며 Functional API에서 구현할 수 없다.

- Functional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다는 장점이 존재한다. Subclassing API는 이러한 Functional API로도 구현할 수 없는 모델들 조차 구현이 가능하다. 하지만, 객체 지향 프로그래밍에 익숙해야 한다. ㅆㄱㄴ

# 9. 다층 퍼셉트론으로 텍스트 분류하기 (MultiLayer Perceptron, MLP)

- 단층 퍼셉트론의 형태에서 은닉층이 1개 이상 추가된 신경망을 다층 퍼셉트론이라고 한다.
- 다층 퍼셉트론은 피드 포워드 신경망의 가장 기본적인 형태로 분류된다.

## 1. 케라스의 texts_to_matrix()

### 1. 정수 인코딩 진행

```python
texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

t = Tokenizer()
t.fit_on_texts(texts)
print(t.word_index)

'''
{'바나나': 1, '먹고': 2, '싶은': 3, '사과': 4, '길고': 5, '노란': 6, '저는': 7, '과일이': 8, '좋아요': 9}
'''
```

### 2. texts_to_matrix()

- 입력된 텍스트 데이터로부터 행렬(matrix)를 만드는 도구
- 해당 함수는 총 4개의 모드를 지원한다. binary, count, freq, tfidf

> count mode : 앞서 배운 문서 단어 행렬 (DTM)을 생성

```python
print(t.texts_to_matrix(texts, mode = 'count'))
'''
[[0. 0. 1. 1. 1. 0. 0. 0. 0. 0.]
 [0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
 [0. 2. 0. 0. 0. 1. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 1. 1.]]
'''
```

- 앞서 배웠듯이 DTM은 bag of words를 기반으로 하므로, 단어 순서 정보는 보존되지 않는다. 구체적으로는 4개의 모든 모드에서 단어 순서 정보는 보존되지 않는다. (?)

> binary : 해당 단어가 존재하는지에 대해서만 관심

```python
print(t.texts_to_matrix(texts, mode = 'binary'))
'''
[[0. 0. 1. 1. 1. 0. 0. 0. 0. 0.]
 [0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 1. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 1. 1.]]
'''
```

> tfidf : TF-IDF 행렬을 만든다. TF를 각 문서에서의 각 단어의 빈도에 자연 로그를 씌우고 1을 더한 값으로 정의..

```python
print(t.texts_to_matrix(texts, mode = 'tfidf').round(2))
'''
[[0.   0.   0.85 0.85 1.1  0.   0.   0.   0.   0.  ]
 [0.   0.85 0.85 0.85 0.   0.   0.   0.   0.   0.  ]
 [0.   1.43 0.   0.   0.   1.1  1.1  0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   1.1  1.1  1.1 ]]
'''
```

> freq : 각 단어의 등장 횟수를 분자로, 각 문서의 크기를 분모로 표현하는 방법이다.

```python
print(t.texts_to_matrix(texts, mode = 'freq').round(2))
'''
[[0.   0.   0.33 0.33 0.33 0.   0.   0.   0.   0.  ]
 [0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.  ]
 [0.   0.5  0.   0.   0.   0.25 0.25 0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.33 0.33 0.33]]
'''
```

## 2. 단계 1. 데이터 이해하기

> 데이터 칼럼 이해하기

```python
print(newsdata.keys())
# dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
```

- data, filenames, target_names, target, DESCR 6개의 열을 가지고 있군..

> 훈련용 샘플 수

```python
print('훈련용 샘플의 개수 : {}'.format(len(newsdata.data)))
# 훈련용 샘플의 개수 : 11314
```

> target_names 칼럼 확인해보기

- target_names에는 20개의 주제의 이름을 담고 있다.

```python
print('총 주제의 개수 : {}'.format(len(newsdata.target_names)))
print(newsdata.target_names)
'''
총 주제의 개수 : 20
['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
'''
```

> 레이블 데이터가 될 target 확인하기

```python
print('첫번째 샘플의 레이블 : {}'.format(newsdata.target[0]))
# 첫번째 샘플의 레이블 : 7
print('7번 레이블이 의미하는 주제 : {}'.format(newsdata.target_names[7]))
# 7번 레이블이 의미하는 주제 : rec.autos
```

- 해당 7의 주제는 target_names[] 안에 숫자를 입력하여 알 수 있다.

> 샘플 데이터 확인해보기

```python
print(newsdata.data[0]) # 첫번째 샘플 출력
```

- 여기까지 확인한 결과 0번 데이터의 레이블은 7이고 rec.autos란 주제를 의미하는 데이터라는 것을 알 수 있다.
- 즉, 이 실습의 목적은 이메일 본문을 보고 20개의 주제 중 어떤 주제인지를 텍스트 데이터 구성만 보고 자동 배치 시켜주는 것이다.

> 데이터 프레임으로 만들기

```python
data = pd.DataFrame(newsdata.data, columns = ['email']) # data로부터 데이터프레임 생성
data['target'] = pd.Series(newsdata.target) # target 열 추가
data[:5] # 상위 5개 행을 출력
```

> 테스트 데이터 만들기

```python
newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) # 'test'를 기재하면 테스트 데이터만 리턴한다.
train_email = data['email'] # 훈련 데이터의 본문 저장
train_label = data['target'] # 훈련 데이터의 레이블 저장
test_email = newsdata_test.data # 테스트 데이터의 본문 저장
test_label = newsdata_test.target # 테스트 데이터의 레이블 저장
```

# 10. 피드 포워드 신경망 언어 모델 (Neural Network Language Model, NNLM)

- 프로그래밍 언어에는 지켜야 하는 문법을 바탕으로 코드를 작성하게 된다.
- 하지만 자연어, 자연어에도 문법이라는 규칙이 존재하기는 하지만, 많은 예외 사항, 시간에 따른 언어의 변화, 중의성과 모호성 문제 등을 전부 명세하기란 어렵다.
- 기계가 자연어를 표현하도록 규칙으로 명세하기가 어려운 상황에서의 대안은 규칙 기반 접근이 아닌, 기계가 주어진 자연어 데이터를 학습하게 하는 것이다.
- 과거에는 기계가 자연어를 학습하게 하는 방법으로 통계적인 접근을 사용했으나, 최근에는 인공 신경망을 사용하는 방법이 자연어 처리에서 더 좋은 성능을 얻고 있다.
- 신경망 언어 모델의 시초인 피드 포워드 신경망 언어 모델에 대해서 배워보자.
- 해당 언어가 제안 되었을 때는 NPLM(Neural Probabilistic Language Model)이라는 이름을 가지고 있었다.

## 1. 기존 N-gram 언어 모델의 한계

- n-gram 언어 모델은 언어 모델링에 바로 앞의 n-1개의 단어만 참고한다.
- n-gramn 언어 모델은 충분한 데이터를 관측하지 못하면 언어를 정확히 모델링하지 못하는 희소 문제(sparsity problem)가 있다.
- $\text{boy is spreading smile}$ 라는 단어 시퀀스가 존재하지 않으면 n-gram 언어 모델에서 해당 단어 시퀀스의 확률 $P(\text{smiles|boy is spreading})$는 0이 되버린다.
- 이는 언어 모델이 예측하기에 boy is spreading 다음에는 smiles이란 단어가 나올 수 없다는 의미이지만, 해당 단어 시퀀스는 현실에서 실제로는 많이 사용되므로 제대로 된 모델링이 아니다.

## 2. 단어의 의미적 유사성

- 희소 문제는 기계가 단어 간 유사도를 알수 있다면 해결할 수 있는 문제이다.

$$P(\text{톺아보다|보도 자료를})\\P(\text{냠냠하다|보도 자료를})$$

- 알다시피 톺아보다라는 단어는 잘 안 사용한다. 즉, $P(\text{톺아보다|보도 자료를})$ 를 0으로 연산하게 되면 예측에 고려할 수 없게 된다.
- 만약 이 언어 모델 또한 단어의 유사도를 학습할 수 있도록 설계한다면, 훈련 코퍼스에 없는 단어 시퀀스에 대한 예측이라도 유사한 단어가 사용된 단어 시퀀스를 참고하여 보다 정확한 예측을 할 수 있게된다.
- 이러한 아이디어를 가지고 탄생한 언어 모델이 신경망 언어 모델 NNLM이다.
- 그리고 단어 간 유사도를 반영한 벡터를 만드는 워드 임베딩의 아이디어이기도 하다.

## 3. 피드 포워드 신경망 언어 모델 (NNLM)

> what will the fat cat sit on

> 원-핫 인코딩

```python
what = [1, 0, 0, 0, 0, 0, 0]
will = [0, 1, 0, 0, 0, 0, 0]
the = [0, 0, 1, 0, 0, 0, 0]
fat = [0, 0, 0, 1, 0, 0, 0]
cat = [0, 0, 0, 0, 1, 0, 0]
sit = [0, 0, 0, 0, 0, 1, 0]
on = [0, 0, 0, 0, 0, 0, 1]
```

- 다음을 학습하는 기계는 what will the fat cat을 입력 받아서 sit을 예측하는데 이 때 기계는 what, will, the, fat, cat의 원-핫 벡터를 입력받아 sit의 웟-핫 벡터를 예측하는 문제가 된다.
- NNLM은 n-gram 언어 모델과 유사하게 다음 단어를 예측할 때, 앞의 모든 단어를 참고하는 것이 아니라 정해진 n개의 단어만을 참고한다.
- n = 4 일 경우, what will the fat cat 라는 단어 시퀀스가 주어졌을 때, 다음 단어를 예측하기 위해 앞의 4개 단어 will the fat cat 까지만 참고하고 그 앞 단어인 what은 무시한다. 이 범위를 윈도우 (window)라고 하기도 한다. 여기서 윈도우의 크기 n은 4이다.

![https://wikidocs.net/images/page/45609/nnlm1.PNG](https://wikidocs.net/images/page/45609/nnlm1.PNG)

- NNLM의 구조를 보자. NNLM은 총 4개의 층(layer)으로 이루어진 인공 신경망이다.
- 4개의 원-핫 벡터를 입력받은 NNLM은 다음층인 투사층(projection layer)을 지나게 된다. 인공 신경망에서 입력층과 출력층 사이의 층은 보통 은닉층이라고 부르는데, 여기서 투사층이 일반 은닉층과 구별되는 특징은 가중치 행렬과의 연산은 이루어지지만, 활성화 함수가 존재하지 않는다는 것이다.
- 투사층의 크기를 M으로 설정하면, 각 입력 단어들은 투사층에서 V x M 크기의 가중치 행렬과 곱해진다.

![https://wikidocs.net/images/page/45609/nnlm2_renew.PNG](https://wikidocs.net/images/page/45609/nnlm2_renew.PNG)

- 위와 같은 모습인데, 식을 보면 알겠듯이, 원-핫 벡터와 가중치 W 행렬의 곱은 사실 W행렬의 i번째 행을 그대로 읽어오는 것과(lookup) 동일하다. 그래서 이 작업을 룩업 테이블(lookup table)이라고 부른다.
- 룩업 테이블 작업을 거치면 V의 차원을 가지는 원-핫 벡터는 이보다 더 차원이 작은 M차원의 단어 벡터로 맵핑된다.
- 해당 결과물로 나오는 벡터는 초기에는 랜덤한 값을 가지지만, 학습 과저에서 값이 계속 변경되는데,이 단어 벡터를 임베딩 벡터 (embedding vector)라고 한다.
- 각 단어가 테이블 룩업을 통해 임베딩 벡터로 변경되고, 투사층에서 모든 임베딩 벡터들의 값은 연결(concatenation)이 된다. **x를 각 단어의 원-핫 벡터,** **NNLM이 예측하고자 하는 단어가 문장에서 t번째 단어**라고 하고, **윈도우의 크기를 n**, **룩업 테이블을 의미하는 함수를 lookup**, **세미콜론(;)을 연결 기호로** 하였을 때, 투사층을 식으로 표현하면 아래와 같다.

$$p^{layer} = (lookup(x_{t-n}); ...; lookup(x_{t-2}); lookup(x_{t-1})) = (e_{t-n}; ...; e_{t-2}; e_{t-1})$$

- 일반적인 은닉층이 활성화 함수를 사용하는 비선형층인 것과는 달리, 투사층은 활성화 함수가 존재하지 않는 선형층이라는 점이 다소 생소하지만, 이 다음부터는 다시 은닉츠을 사용하므로, 일반적인 피드 포워드 신경망과 동일해진다.

![https://wikidocs.net/images/page/45609/nnlm4.PNG](https://wikidocs.net/images/page/45609/nnlm4.PNG)

- 투사층의 결과는 h의 크기를 가지는 은닉층을 지난다. 일반적인 피드 포워드 신경망에서 은닉층을 지난다는 것은 은닉층의 입력은 가중치가 곱해진 후 편향이 던해져 활성화 함수의 입력이 된다는 의미이다.
- NNLM은 예측값과 실제값의 벡터가 가까워지게 되게 하기 위해서, 손실 함수로 cross-entropy 함수를 사용한다. 그리고 역전파가 이루어지면서 가중치 행렬들이 학습되는데, 이 과정에서 임베딩 벡터값들도 학습이 된다.
- NNLM으로 얻을 수 있는 이점은 충분한 양의 훈련 코퍼스를 가지고 있다면 결과적으로 수많은 문장에서 유사한 목적으로 사용되는 단어들은 결국 유사한 임베딩 벡터값을 얻게되는 것에 있다.
- 이렇게 되면 훈련이 끝난 후 다음 단어를 예측 과정에서 훈련 코퍼스에서 없던 단어 시퀀스라고 하더라도 다음 단어를 선택할 수 있다.

## 4. NNLM의 이점과 한계

### 1. 기존 모델에서의 개선점

- 밀집 벡터를 이용하여 단어의 유사도를 표현, 희소 문제를 해결,

### 2. 고정된 길이의 입력 (Fixed-length input)

- NNLM 또한 한계가 존재한다. 그것은 NNLM도 정해진 n개의 단어만을 참고한다는 점인데, 이는 버려지는 단어들이 가진 문맥 정보는 참고할 수 없다는 것을 의미한다.
