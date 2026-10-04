# Local Likelihood

## Local Likelihood and Other Models

어떤 모수모형이더라도 local하게 만들 수 있다.(관측치의 웨이트를 포함하는 경우라면 모두)

likelihood의 개념자체가 그럴듯한 모수를 추정해내는 개념이다. 그렇기 때문에 모수모형이라는 이야기가 나온다. Local likelihood를 이용하면 모수모형이더라도 Maximum Likelihood로 로컬하게 추정할 수 있다. 모수모형은 global하게 피팅하는게 국룰이고, semi모수나 비모수 모형의 경우가 로컬한 모형이라는 인식이 있는데 모수에 대해 해석할 수 있는 모수모형 역시 로컬하게 피팅할 수 있다는 점에서 장점이 있다.

실제로 logistic regression을 생각해보자.

$$
Y_i|X \sim Ber(logit(\eta(X)) \\
\eta(X)=\beta_0+\beta_1x+\cdots+\beta_px^p
$$

x에 대한 $$\beta$$ local log-likelihood를 생각하면 다음과 같다.&#x20;

$$
\begin{split}
l(\beta) & =\sum^n_{i=1}\{Y_ilog(logit(\eta(X))+(1-Y_i)log(1-logit(\eta(X))\}  \\
& = \sum^n_{i=1}l(Y_i,\eta(X))
\end{split}
$$

$$
l_{x,h}(\beta)=\Sigma^n_{i=1}l(Y_i,\eta(X))K_h(x-X)
$$

\[ref: [https://bookdown.org/egarpor/NP-UC3M/kre-ii-loclik.html](https://bookdown.org/egarpor/NP-UC3M/kre-ii-loclik.html)]

Y가 binary 응답일 때 경향성을 시각화하기에 좋다.

![](../../.gitbook/assets/ch6\_3.png)

Y가 범주형인 경우에 모형의 시각적 해석을 붙이기가 어려웠는데 local linear logistic은 이러한 해석이 가능하다. 위 그림은 실제 특정 X값에서의 확률이 예측된 것을 그린 것이다.
