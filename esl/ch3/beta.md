# Coefficients(Beta)

## Dive into$$\hat{\beta}$$&#x20;

&#x20;   $$\hat{\beta}^{LS}$$contains $$\mathbf{y}$$. When we assume the error follows probability distribution, $$\mathbf{y}$$also becomes random variable that has uncertainty. Thus $$\hat{\beta}^{LS}$$ also follows some distribution related to the distribution of error.

**Don't get confused!** In a frequentist view, $$\beta$$ is constant. However the estimation value of beta$$\hat{\beta}=f\{(X_1,Y_1),...,(X_n,Y_n)\}$$ is a statistic so it has a distribution.

$$
\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}  =(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{X}\beta+\epsilon)
\\\hat{\beta}\sim N(\beta,(\mathbf{X}^T\mathbf{X})^{-1}\sigma^2)
$$

$$
\hat{\sigma}^2=\dfrac{1}{N-p-1}\sum^N_{i=1}(y_i-\hat{y}_i)^2 \\
(N-p-1)\hat{\sigma}^2 \sim \sigma^2\chi^2_{N-p-1}
$$

Square of Normal becomes chi-square.

$$
z_j=\dfrac{\hat{\beta}_j}{\hat{sd}(\hat{\beta}_j)} =\dfrac{\hat{\beta}_j}{\hat{\sigma}\sqrt{v_j}} \sim t(df) \quad s.t. \; N-p-1 = df
$$

$$\hat{Var}(\hat{\beta})=(X^TX)^{-1}\hat{\sigma}^2, \; \hat{Var}(\hat{\beta}_j)=v_j\hat{\sigma}^2\\v_j=j_{th} \; diagonal \; element \; of \; (X^TX)^{-1}$$

&#x20;   Now we know the distribution of test statistic $$z_j$$, so we can test whether the coefficient is zero and get the confidence interval. When we want to test whether subset of coefficients is zero, we can use the test statistic below.

$$
F=\dfrac{among \; group \; var}{within \; group \; var}=\dfrac{MSR}{MSE}=\dfrac{(RSS_0-RSS_1)/(p_1-p_0)}{RSS_1/(N-p_1-1)}
$$

&#x20;   $$F$$ has a distribution, so we can do zero value test for the coefficient. This testing gives hint for eliminating some input variables.

****

## **Gauss-Markov Theorem**

&#x20;This Theorem says Least Square estimates are good! There are three assumptions below.

1. Input variables are fixed constant.
2. $$E(\varepsilon_i)=0$$
3. $$Var(\varepsilon_i)=\sigma^2<\infty, \quad Cov(\varepsilon_i,\varepsilon_j)=0$$

Under these assumptions, OLS is the best estimate by GM.(Refer to statkwon.github.io)

$$
E(\hat{\beta})=E(\tilde{\beta})=\beta
\\ Var(\tilde{\beta})- Var(\hat{\beta}) \;: positive \; semi-definite
$$

**Proof**

$$\tilde{\beta}=Cy, \; C=(X^TX)^{-1}X+D, \; D: \; K\times n \; matrix$$

$$
{\displaystyle {\begin{aligned}\operatorname {E} \left[{\tilde {\beta }}\right]&=\operatorname {E} [Cy]\\&=\operatorname {E} \left[\left((X'X)^{-1}X'+D\right)(X\beta +\varepsilon )\right]\\&=\left((X'X)^{-1}X'+D\right)X\beta +\left((X'X)^{-1}X'+D\right)\operatorname {E} [\varepsilon ]\\&=\left((X'X)^{-1}X'+D\right)X\beta \quad \quad (\operatorname {E} [\varepsilon ]=0)\\&=(X'X)^{-1}X'X\beta +DX\beta \\&=(I_{K}+DX)\beta .\\\end{aligned}}}
$$

$$
{\displaystyle {\begin{aligned}\operatorname {Var} \left({\tilde {\beta }}\right)&=\operatorname {Var} (Cy)\\&=C{\text{ Var}}(y)C'\\&=\sigma ^{2}CC'\\&=\sigma ^{2}\left((X'X)^{-1}X'+D\right)\left(X(X'X)^{-1}+D'\right)\\&=\sigma ^{2}\left((X'X)^{-1}X'X(X'X)^{-1}+(X'X)^{-1}X'D'+DX(X'X)^{-1}+DD'\right)\\&=\sigma ^{2}(X'X)^{-1}+\sigma ^{2}(X'X)^{-1}(DX)'+\sigma ^{2}DX(X'X)^{-1}+\sigma ^{2}DD'\\&=\sigma ^{2}(X'X)^{-1}+\sigma ^{2}DD' \quad \quad (DX=0)\\&=\operatorname {Var} \left({\widehat {\beta }}\right)+\sigma ^{2}DD' \quad \quad (\sigma ^{2}(X'X)^{-1}=\operatorname {Var} \left({\widehat {\beta }}\right))\end{aligned}}}
$$

&#x20;  $$DD'$$is a positive semi-definite matrix.($$\because$$it is a symmetric matrix.)$$\hat{\beta}^{LS}$$is MVUE(Minimum Variance Unbiased Estimator).



**Always good?**

$$
\begin{split}
  Err(x_0) ={}& E[(Y-\hat{f}(x_0))^2|X=x_0] \\
     = \; & \sigma^2_\epsilon+[E\hat{f}(x_0)-f(x_0)]^2+E[\hat{f}(x_0)-E\hat{f}(x_0)]^2 \\ 
= \; & \sigma^2_\epsilon+Bias^2(\hat{f}(x_0))+Var(\hat{f}(x_0)) \\
= \; & Irreducible \; Error +Bias^2+Variance
\end{split}
$$

&#x20;   We can image the biased estimator away from old school OLS. By keeping more bias, we can lower much more variance. It means we more accurately predict future value.



**Ridge, Lasso, and Elastic Net**

![](<../../.gitbook/assets/image (51).png>)





##
