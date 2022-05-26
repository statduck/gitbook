# Natural Cubic Spline

&#x20;To overcome the weakness of local polynomial regression, natural cubic spline appears. This model adds linear constraint on the border line.

&#x20;To add this constraint, we need to think about this equation.

$$
f(X)=\beta_1+\beta_2X+\beta_3(d_1(X)-d_{K-1}(X))+\cdots+\beta_K(d_K(X)-d_{K-1}(X))
$$

$$
d_k(X)=\dfrac{(X-\xi_k)^3_+-(X-\xi_K)^3_+}{\xi_K-\xi_k}
$$

Proof: [https://statkwon.github.io/ml/natural-spline/](https://statkwon.github.io/ml/natural-spline/)

