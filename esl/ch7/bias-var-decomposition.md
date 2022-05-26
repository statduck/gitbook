# Bias-Var decomposition

## The Bias-Variance Decomposition

&#x20;Let's think about one specific point. To deal with the relationship between bias and variance, we have to decompose a test error.&#x20;

### Regression

$$(Y=f(X)+\epsilon)$$**, error assumption**$$(E(\epsilon)=0, Var(\epsilon)=\sigma^2_\epsilon)$$, Squared-error loss, Regression fit $$\hat{f}(X)$$

$$
\begin{split}
  Err(x_0) ={}& E[(Y-\hat{f}(x_0))^2|X=x_0] \\
     = \; & \sigma^2_\epsilon+[E\hat{f}(x_0)-f(x_0)]^2+E[\hat{f}(x_0)-E\hat{f}(x_0)]^2 \\ 
= \; & \sigma^2_\epsilon+Bias^2(\hat{f}(x_0))+Var(\hat{f}(x_0)) \\
= \; & Irreducible \; Error +Bias^2+Variance
\end{split}
$$

&#x20;

### k-nearest-neighbor regression fit

$$
\begin{split}
  Err(x_0) ={} & E[(Y-\hat{f}_k(x_0))^2|X=x_0] \\
     = \; &\sigma^2_\epsilon+[f(x_0)-\dfrac{1}{k}\Sigma^k_{l=1} f(x_{(l)})]^2+\dfrac{\sigma^2_\epsilon}{k} \\
\end{split}
$$

$$
Var(\hat{f}_k(x_0))=\frac{1}{k^2}k\sigma^2_\epsilon
$$

### Linear model

$$
\begin{split}
  Err(x_0) ={} & E[(Y-\hat{f}_p(x_0))^2|X=x_0] \\
     = \; &\sigma^2_\epsilon+[f(x_0)-E\hat{f}_p(x_0)]^2+||h(x_0)||^2\sigma^2_\epsilon
\end{split}
$$

Here $$h(x_0)=X(X^TX)^{-1}x_0$$

$$
Var(\hat{f}(x_0)) = Var(x_0^T\beta)= Var(x_0^T(X^TX)^{-1}X^Ty)=x_0^T(X^TX)^{-1}x_0\sigma^2_\epsilon=p\sigma^2_\epsilon
$$

&#x20;Let's look at In-sample error. This error is the error that X is equal to the value in training set, but y is just a random quantity.

$$
\dfrac{1}{N}\sum^N_{i=1}Err(x_i)=\sigma^2_\epsilon+\dfrac{1}{N}\sum^N_{i=1}[f(x_i)-E\hat{f}(x_i)]^2+\dfrac{p}{N}\sigma^2_\epsilon \\
\sum^N_{i=1}x_i^T(X^TX)^{-1}x_i=tr(X(X^TX)^{-1}X^T)=tr((X^TX)^{-1}X^TX)=tr(I_p)=p
$$

It means **In-sample error is affected by the dimension of input space.**

### Ridge regression

&#x20;Bias can be composed into things in linear model.

$$
\begin{split}
  E_{x_0}[f(x_0)-E\hat{f}_\alpha(x_0)] ={} & E_{x_0}[f(x_0)-x_0^T\beta_*]^2+E_{x_0}[x_0^T\beta_*-Ex^T_0-Ex_0^T \hat{\beta}_\alpha ]^2\\
     = \; &Ave[Model \; Bias]^2+Ave[Estimation \; Bias]^2
\end{split}
$$

In short, by restricting the range of parameters the bias is increased than one of least square model. However, the variance would be reduced due to this increased bias. In error decomposition bias has the squared form so, a slight increase in bias can decrease in variance.



![](<../../.gitbook/assets/image (2).png>)
