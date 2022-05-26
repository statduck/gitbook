# Optimism

## Optimism of the Training Error Rate

$$
Err_{in}=\dfrac{1}{N}\sum^N_{i=1}E_{Y_0}[L(Y_i^0,\hat{f}(x_i))|\mathcal{T}]
$$

The $$Y^0$$notation indicates that we observe N new response values at each of the training points $$x_i, i=1,2,\dots,N$$.

$$
Err_{extra}=Err_{in}+Err_{out\;of}=(\bar{err}+op)+Err_{out\;of}
$$

&#x20;Extra sample error can be decomposed into in-sample error and out of sample error. In sample error is the sum of training error and optimism.

$$
op\equiv Err_{in}-\bar{err}
$$

&#x20;(Because $$Err_{in}$$and $$op$$are random quantity, we use triple equal sign.) It means $$op$$has the distribution same as $$Err_{in}-\bar{err}$$. Let's get the expected op to predict it.

$$
\omega\equiv E_y(op)
$$

&#x20;More concisely, in a right side this is conditioned on $$\mathcal{T}$$ but we can just use y(X is given input, we only need to consider random quantity y). In several loss functions like squared error and 0-1 $$\omega$$satisfies the following equation.&#x20;

$$
\omega=\dfrac{2}{N}\sum^N_{i=1}Cov(\hat{y}_i,y_i)
$$

> Proof: [https://stats.stackexchange.com/questions/88912/optimism-bias-estimates-of-prediction-error](https://stats.stackexchange.com/questions/88912/optimism-bias-estimates-of-prediction-error)

$$
Cov(\hat{Y},Y)=Cov(\hat{Y},\hat{Y}+\epsilon)=Cov(\hat{Y})=Cov(HY)=HCov(Y)H^T \\
Cov(\hat{y}_i,y_i)=[HH^T]_{ii}\sigma^2 \\
\sum^N_{i=1}Cov(\hat{y}_i,y_i)=\sum[X(X^TX)^{-1}X^T]_{ii}\sigma^2=trace(X)\sigma^2=d\sigma^2
$$

$$
E_y(Err_{in})=E_y(\bar{err})+\frac{2}{N}\sum Cov(\hat{y}_i,y_i)=E_y(\bar{err})+2*\frac{d}{N}\sigma^2_\epsilon
$$

Expected in-sample error. Trace becomes d because of effective number of parameters.

> An obvious way to estimate prediction error is to estimate the optimism and then add it to the training error.

&#x20;To estimate In-sample error we estimate expected In-sample error



### Estimates of In-sample Prediction Error

$$
\hat{Err_{in}}=\bar{err}+\hat{w}
$$

In-sample error estimation = $$\omega$$(average optimism) estimation.

The way omega is estimated decide following equations:



#### Cp

$$
C_p=\bar{err}+2*\dfrac{d}{N}\hat{\sigma}^2_\epsilon
$$

#### &#x20;AIC

AIC(Akaike Information Criterion) uses log-likelihood loss function. 아래의 식은 Expected in-sample error에 관한 식이다.&#x20;

$$
-2*E[logPr_\hat{\theta}(Y)] \sim -\frac{2}{N}E[loglik]+2\frac{d}{N} \\
loglik=\Sigma logPr_{\hat{\theta}}y_o
$$





> Proof: [http://faculty.washington.edu/yenchic/19A\_stat535/Lec7\_model.pdf](http://faculty.washington.edu/yenchic/19A\_stat535/Lec7\_model.pdf)

The idea of AIC is to adjust the empirical risk to be an unbiased estimator of the true risk in a parametric model. Under a likelihood framework, the loss function is the negative log-likelihood function



Logistic regression model(binomial log likelihood)

$$
AIC=-\frac{2}{N}loglik+2\frac{d}{N}
$$

Gaussian model

$$
AIC=\bar{err}+2\frac{d}{N}\hat{\sigma}^2_\epsilon
$$

With tuning parameter $$\alpha$$

$$
AIC(\alpha)=\bar{err}(\alpha)+2\frac{d(\alpha)}{N}\hat{\sigma}^2_\epsilon
$$

&#x20;The aim is to find alpha that minimizes the value above using our estimated value of test error.
