# Bayesian Regression

Compare Aerobic group and Running group! We can compare these two groups by showing that $$\beta_0=\beta_1=0$$. Our concern is **Full probability model** $$p(y,\beta,\sigma^2)=p(\beta,\sigma^2)p(y|\beta,\sigma^2)$$

| semi-conjugate prior                                    | full-conjugate prior |
| ------------------------------------------------------- | -------------------- |
| Prior and conditional probability has same distribution | Dependent prior      |

### Semi conjugate prior

likelihood: $$y|X,\beta,\sigma^2 \sim MVN(X\beta,\sigma^2I)$$

prior:

posterior:

$$
\begin{split}
p(\beta|y,\sigma^2) \propto {} & p(y|\beta,\sigma^2)p(\beta) \\
\propto \; & exp(-\frac{1}{2\sigma^2}(y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta)-\frac{1}{2}(\beta^T\Sigma^{-1}_0\beta-2\beta^T\Sigma^{-1}_0\beta_0)) \\
\propto \; & exp(-\frac{1}{2}\beta^T(X^TX/\sigma^2+\Sigma^{-1}_0)\beta+\beta^T(X^Ty/\sigma^2+\Sigma^{-1}_0\beta_0) 
\end{split}
$$



### Assignment

### MCMC

Markov Chain Monte Carlo

Week5 이전까지는 Exact Bayesian Inference Analytic form을 구하는 방식을 주로 다루었다. 그렇지만 Conjuate한 상황에서도 시뮬레이션으로 더 간단하게 풀 수 있고 non-conjugate한 경우에도 시뮬레이션을 활용할 수 있다. 이러한 시뮬레이션 방식에는 다음과 같이 크게 두 가지 접근법이 존재한다.

#### Marginalization

MC (Independent Monte Carlo)

MCMC (Markov Chain Monte Carlo) Metropolis Hastings

* Special Case (Gibbs sampling)

#### Optimization

Variational Inference 최적의 q를 찾는 것이다.

$$
q(\theta|\eta) \sim p(\theta|Data)
$$

최적화 방법론은 추후에 알아보고, 우선은 Marginalization에 대해 먼저 살펴보자.

### Monte Carlo Method

현재 우리의 목표는 **사후분포 추정**이다. **사전분포**는 우리가 가정하는 것이고 데이터가 **특정분포**에서 뽑힌다고 가정하면 likelihood도 쉽게 구할 수 있다. 그리고 이 둘의 관계를 이용해 사후분포를 구하는 게 핵심이다.

$$
p(\theta|Data)=\dfrac{p(Data|\theta)p(\theta)}{\int p(Data|\theta)p(\theta)d\theta}
$$

분모(Evidence)는 **Prior Belief \* Likelihood**의 적분형태이다. Closed form을 구할 수 없는 경우가 많으므로 가장 단순한 방식으로 격자식 적분이 가능하다. **Prior Belief \* Likelihood**에서 각각을 이산형으로 바라보고 계산하는 방식이다. 그렇지만 파라미터의 수가 증가할 때마다 계산해야하는 값이 지수적으로 증가하므로 계산소모적이고 차원의 저주에 빠지기 쉽다는 단점이 있다. 또 간지가 안난다!

### Monte Carlo integration (random)

결국 이는 적분을 근사적으로 할건데 리만 적분과 유사한 형태로 시그마를 취하는게 아니라 다른 효율적 방식에 대한 고민을 해보자는 것이다. 아까 분모에 있던 Evidence를 구하기 위한 첫번째 시도는 적분을 평균의 관점으로 보는 것이다.

그렇다면 다음과 같이 Law of Large Number을 사용할 수 있다(통계량의 Consistency 확보)

Let $$g(\theta)$$ be any (computable) function

$$
\frac{1}{s}\sum^S_{s=1}g(\theta^s) \xrightarrow{} E[g(\theta)|y_1,\dots,y_n] = \int g(\theta)p(\theta|y_1,\dots,y_n)d\theta, as \; S \xrightarrow\infty \; by \; LLN
$$

그리고 이를 이용

$$
E(g(\theta)|y_1,\dots,y_n)=\int g(\theta)p(\theta|y_1,\dots,y_n)d\theta) \sim \frac{1}{s} \sum^S_{s=1} g(\theta^s)
$$

\$$g(\theta$)$의 정보를 얻기 위해서 통계량을 이용하는 모습이 익숙하다.

$$
\theta_d^{(s)}: deterministic \; sample \; points
$$

그렇다면 샘플을 얼마나 뽑아야하는가?에 대한 문제가 생긴다. 이 때 Sample Size와 Precision의 관계를 나타내주는 CLT를 이용해야한다.

실제 데이터를 이용한다고 가정하자. known variance 가정을 하자.

$$
\theta \sim N(\mu, \tau^2) \\
{y_1,\dots,y_n|\theta} \sim N(\theta,\sigma^2)
$$

$$
\theta|D \sim (\mu_n, \tau^2_n) \\
\mu_n = \bar{y}\frac{n/\sigma^2}{n/\sigma^2+1/\tau^2}+\mu\frac{1/\tau^2}{n/\sigma^2+1/\tau^2} \\
\tau^2_n = 1/(n/\sigma^2 + 1/\tau^2)
$$

이 경우 우리가 prior와 data에 대한 정보를 준다면 쉽게 사후분포를 구할 수 있다. 하지만 이렇게 analytic form이 아니라 Metropolis algorithm을 이용한 시뮬레이션 방식을 생각해보자.

$$
r=\frac{p(\theta^*|y)}{p(\theta^{(s)}|y)}=\frac{\Pi^n_{i=1} dnorm(y_i,\theta^*,\sigma)}{\Pi^n_{i=1}dnorm(y_i,\theta^{(s)},\sigma)}\times\frac{dnorm(\theta^*,\mu,\tau)}{dnorm(\theta^{(s)},\mu,\tau)}
$$

$$
logr = \sum^n_{i=1}[log dnomr(y_i,\theta^*,\sigma)-log dnorm(y_i,\theta^{(s)},\sigma)]+log dnorm(\theta^*,\mu,tau)-log dnorm(\theta^{(s)},\mu,tau)
$$

#### Metropolis-Hastings algorithm

$$log\mu < log r$$인 경우에 accept 된다.



### Assignment

```
## MCMC
s2<-1 ; t2<-10 ; mu<-5 
y<-c(9.37, 10.18, 9.16, 11.60, 10.33) #data
theta<-0 ; delta<-2 ; S<-10000 ; THETA<-NULL ; set.seed(1)

sampling <- function(S){
  for(s in 1:S){
    theta.star<-rnorm(1,theta,sqrt(delta))
    log.r<-( sum(dnorm(y,theta.star,sqrt(s2),log=TRUE)) +
               dnorm(theta.star,mu,sqrt(t2),log=TRUE) )  -
      ( sum(dnorm(y,theta,sqrt(s2),log=TRUE)) +
          dnorm(theta,mu,sqrt(t2),log=TRUE) ) 
    if(log(runif(1))<log.r) { theta<-theta.star }
      THETA<-c(THETA,theta)
  }
  return(THETA)
}

#### Figure 10.3
library(coda)
set1 <- mcmc(sampling(1000)); set2 <- mcmc(sampling(10000))
summary(set1); summary(set2)
plot(set1)
plot(set2)
autocorr.plot(set1)
autocorr.plot(set2)

# gelman.diag(set1)
# gelman.plot(set1)
```

![](<../.gitbook/assets/스크린샷 2021-05-18 오후 10.30.23.png>)

![](<../.gitbook/assets/image (7).png>)

![](<../.gitbook/assets/image (8).png>)

![](<../.gitbook/assets/image (9).png>)

![](<../.gitbook/assets/image (10).png>)



#### Prove that $$\beta|y,\sigma^2 \sim N(\frac{g}{g+1}\hat{\beta}{mle},\frac{g}{g+1}Var(\hat{\beta}_{mle}))$$

We already showed that

$$
\beta|y,\sigma^2 \sim N(\beta_n, \Sigma_n)
$$

$$
\beta_n=\Sigma_n(X^Ty/\sigma^2+\Sigma^{-1}_0\beta_0), \; \Sigma^{-1}_n=X^TX/\sigma^2 + \Sigma^{-1}_0
$$

under the full conjugate prior setting,

$$
\beta_0=0, \Sigma_0=g\sigma^2(X^TX)^{-1}
$$

Let's plugging in!

$$
\Sigma_n^{-1}=\frac{X^TX}{\sigma^2}+\frac{X^TX}{g\sigma^2}=((\frac{g+1}{g})\frac{X^TX}{\sigma^2})
$$

$$
\hat{\beta}_{mle}=(X^TX)^{-1}X^Ty \\
Var(\hat{\beta}_{mle})=(X^TX)^{-1}X^TX(X^TX)^{-1}\sigma^2=(X^TX)^{-1}\sigma^2
$$

$$
\Sigma_n=\frac{g}{g+1}(X^TX)^{-1}\sigma^2=\frac{g}{g+1}Var(\hat{\beta}_{mle}) \\
\beta_n=\frac{g}{g+1}(X^X)^{-1}\sigma^2(\frac{X^Ty}{\sigma^2})=\frac{g}{g+1}\hat{\beta}_{mle}
$$

####

#### Prove that $$SSR(g) \xrightarrow{\infty} SSR(\hat{\beta}_{mle})$$



$$
\begin{split}
SSR(\hat{\beta}_{mle}) = {} & y^Ty-(X\beta)^T(X\beta) \\
\; = & y^Ty-(X(X^TX)^{-1}X^Ty)^T(X(X^TX)^{-1}X^Ty) \\
\; = & y^Ty-(y^TX(X^TX)^{-1}X^TX(X^TX){-1}X^Ty) \\
\; = & y^T{I-X(X^TX)^{-1}X^T}y \\
SSR(g) = &y^T(I-\frac{g}{g+1}X(X^TX)^{-1}X^T)y
\end{split}
$$

$$
\therefore SSR(g) \xrightarrow{} SSR(\hat{\beta}_{mle})  \quad (\because lim_{g\xrightarrow{}\infty}\frac{g}{g+1}=1)
$$

