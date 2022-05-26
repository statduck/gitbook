# Session 9(editing)

변수선택

Freq approach

* T-test: 여러 계수들을 함께 비교하기에 적절치 않음(해결: Bonferroni, Tukey): $$(0.95)^p$$
* AIC: $$-2ln(L)+2k, \quad k=\# \;of \; param$$

Bayesian approach

각각의 모델을 확률을 가지고 있는 확률변수로 바라본다.

$$
P(model1 | data)=a \quad vs \quad P(model2 | data)=b
$$

두 가지를 비교&#x20;



Indicator Variable 사용.



$$
\beta_k=z_k*b_k \\
Data \; Model:y_i=z_1b_1x_{i,1}+\cdots+z_pb_px_{i,p}+\epsilon_i \\
Prob \; Model:P(y,\beta,\sigma^2,z)=p(z)p(\sigma^2)p(\beta|\sigma^2,z)p(y|\beta,\sigma^2)
$$

화살표 간 부분이 조건으로 걸리는 것으로 보자.



z가 어떤 값을 가지는지에 따라 z와 model을 일대일로 대응시킬 수 있다.

$$
Y_i=\beta_1x_{i,1}+\beta_2x_{i,2}+\beta_3x_{i,3}+\beta_4x_{i,4} \\
E(Y|x,b,z=(1,0,1,0))=b_1x_1+b_3x_3 \\
E(Y|x,b,z=(1,1,0,0))=b_1x_1+b_2x_2 \\
E(Y|x,b,z=(1,1,1,0))=b_1x_1+b_2x_2+b_3x_3
$$

$$
P(z|X,Y) \propto P(Y|X,z)P(z)
$$

P(z): 각 Model에 Prior를 부여

P(Y|X,z): 각 모델에 대하여 데이터가 말하는 Likelihood를 계산한다.



Parameter의 개수가 많지 않은 경우에 모델의 사후분포를 직접 구해낼 수 있다. 가장 확률이 높은 모델을 선택하면 된다.



$$
P(z|X,Y)=\frac{p(z)p(y|X,z)}{\Sigma p(\tilde{z})p(y|X,\tilde{Z})}
$$

분모항의 경우 파라미터 개수가 적은 경우 좌변을 구할 수 있지만 파라미터의 개수가 많으면 계산이 힘들다. 따라서 분모항을 계산하지 않고도 모델을 비교하는 방법을 사용할것이다.

$$
odds(z_a,z_b|y,X)=\frac{p(z_a|y,X)}{p(z_b|y,X)}=\frac{p(z_a)}{p(z_b)}*\frac{p(y|X,z_a)}{p(y|X,z_b)}
$$

해당 방식을 다음과 같이 정리할 수 있다. posterior odds = prior odds \* "Bayes factor"\


이 경우 $$p(z)$$는 알고 있고, $$p(Y|X,z)$$의 경우 베타의 사전분포가 g-prior이면 구할 수있다.

$$
\beta_0=0, \Sigma_0=g\sigma^2(X^TX)^{-1} \\
\beta|y,\sigma^2 \sim N(\frac{g}{g+1}\hat{\beta}_{mle},\frac{g}{g+1}Var(\hat{\beta}_{mle}))
$$

g가 커지면 prior가 의미가 없어졌음.





$$
\begin{split}
p(y|X,z)=\iint p(y,\beta,\sigma^2|X,z)d\beta d\sigma^2 \\

\end{split}
$$

