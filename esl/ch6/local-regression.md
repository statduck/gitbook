# Local regression

## ✏️ local regression

```
locality is given by kernel method.
  1) Directely use kernel in estimating function form.
  2) We already know the function form, so use kernel in estimating parameters.
```

The local linear, polynomial uses 2)

### local linear regression

$$
\hat{f}(x_0)=\hat{\alpha}(x_0)+\hat{\beta}(x_0)x_0 \\
\min_{\alpha(x_0),\beta(x_0)}\Sigma^N_{i=1}K_\lambda(x_0,x_i)[y_i-\alpha(x_o)-\beta(x_0)x_i]^2
$$

Orginally, linear regression estimates function globally. We can make a variation using kernel in doing regression. Each function value has different kernel value, so we can estimate function locally.

$$
\hat{f}(x_0)=b(x_0)^T(B^TW(x_0)B)^{-1}B^{T}W(x_0)y=\Sigma l_i(x_0)y_i
$$

and then, what is a local estimation? It means we can derive $$\hat{f}(x_0)$$after we fit the regression line. When fitting the line, we consider the density of distance so closer points get more probability. Closer points have more power on determining the beta coefficients.



![](../../.gitbook/assets/ch6\_2.png)

However, kernel method has a fatal flaw. In the side data, we don't consider the half of data. We just care about points in the right of one point which is the located on the leftmost. We call it **boundary effect.**&#x20;

> Trevor Hastie, Clive Loader "Local Regression: Automatic Kernel Carpentry," Statistical Science, Statist. Sci. 8(2), 120-129, (May, 1993)

In this paper, local linear regression solves this boundary issue like the below.&#x20;

$$
\begin{split}
 E\hat{f}(x_0)= & \sum^N_{i=1}l_i(x_0)f(x_i) \\= & f(x_0)\sum^N_{i=1} l_i(x_0)+f'(x_0)\sum^N_{i=1} (x_i-x_0)l_i(x_0) \\ & + \dfrac{f''(x_0)}{2} \sum^N_{i=1}(x_i-x_0)^2 l_i(x_0) + R  \end{split}
$$

This expression is subject to $$\Sigma l_i(x_0) =1$$, $$\Sigma (x_i-x_0)l_i(x_0)=0$$. It means  bias$$E(\hat{f}(x_0))-f(x_0)$$ only depends on quadratic and higher order terms(curvature term). So if we estimate function in first order, curvature term and R becomes 0(bias becomes bigger). It solves the boundary issue.&#x20;

**Q. proof: statkwon.github.io**

To sum up, to control boundary effect we need to make a model with slope. Local linear regression is best for it.

### local polynomial regression

$$
\hat{f}(x_0)=\hat{\alpha}(x_0)+\Sigma^d_{j=1}\hat{\beta}_j(x_0)x_0^j
$$

$$
\min_{\alpha(x_0),\beta(x_0),j=1,...,d}\Sigma^N_{i=1}K_\lambda(x_0,x_i)[y_i-\alpha(x_o)-\Sigma^d_{j=1}\beta_j(x_0)x_i^j]^2
$$

Local polynomial can also derive $$\hat{f}$$ as local linear regression does. The difference is how to define  $$b(x_0)^T$$.&#x20;

* Linear  $$[1 \quad x]$$&#x20;
* Polynomial $$[1 \quad x \quad x^2 \quad ...]$$

## ✏️ Local Regression in p dimension

$$
\min_{\beta(x_0)}\Sigma^N_{i=1} K_\lambda (x_0,x_i)(y_i-b(x_i)^T\beta(x_0))^2
$$

$$
K_\lambda(x_0,x)=D(\dfrac{||x-x_0||}{\lambda})
$$

Let's think about high dimension. $$||x-x_0||$$ is a euclidean distance, and it is needed to standardize this with unit standard deviation because a different x has different distance.&#x20;

&#x20;Boundary effects become serious in high dimension, because of the dimensionality curse. The number of data close to boundary increase in high dimension, and also it becomes hard to visualize our data even if smoothing is usually for the visualization. Sampling density is proportional to $$N^{1/p}$$, so sparsity problem would happen.



### Structured Kernels

각 축에 동일한 웨이트를 적용하지 않고, 좌표축을 표준화 하는 대신 weight matrix A를 곱해준다. $$K_{\lambda,A}(x_0,x)=D(\dfrac{(x-x_0)^T A(x-x_0)}{\lambda})$$

Diagonal condition: 각 $$X_j$$의 영향을 조절할 수 있다.

* A가 High-frequency contrasts에 덜 집중하도록 만들기.
* low rank A가정 - ridge function을 만든다.

그렇지만 이렇게 A의 차원이 커지면 다루기 힘들어져 Structured regression을 사용한다.

### Structured Regression

ANOVA decomposition의 방식을 사용. $$f(X)=\alpha(Z)+\beta_1(Z)X_1+\cdots+\beta_q(Z)X_q$$

##
