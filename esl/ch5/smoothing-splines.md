# Smoothing Splines

## ✏️ Smoothing Spline

&#x20;**Avoiding the knot selection problem completely by using a maximal set of knots.**

$$
RSS(f,\lambda)=\sum^N_{i=1}\{y_i-f(x_i)\}^2+\lambda\int \{f''(t)\}^2dt
$$

&#x20;Our goal is to find the form of function minimizing RSS. The constrains mean curvature as follows:

$$
r = (x,y), ||r'||=\sqrt{x'(s)^2+y'(s)^2}=1 \\
T(s)=(x'(s),y'(s)) = unit \; tangent \; vector \\
\kappa(s)=||T'(s)||=||r''(s)||=\sqrt{x''(s)^2+y''(s)^2}
$$

$$
x=t, y=f(t), \kappa(s)=\sqrt{f''(s)^2}=|f''(s)|
$$

&#x20;To get the curvature on whole points, we calculate$$\int |f''(s)| ds$$. For a convenience of calculation, we can adjust it into $$\int f''(s)^2 ds$$. The curvature on one point is the norm of the derivative of a tangent vector, and $$\kappa(s)$$ means the curvature on this point. What we want to get is the norm of curvature on support range. In this function, the norm of curvature becomes small when this function becomes smooth(close to linear).

{% hint style="info" %}
First term controls error, and second term controls curvature.
{% endhint %}

&#x20;Lambda is a fixed smoothing parameter). To control the curvature, the error term can be big, so we need to control lambda for balancing. When lambda becomes so big, rss cane reduced a lot when we make the curvature a bit small. The bigger lambda becomes, the smoother function becomes.

* lambda = 0: f can be any function that interpolates the data.
* lambda = inf: the simple least squares line fit, since no second derivative can be tolerated.



{% hint style="info" %}
&#x20;There must exist the second derivative form of the function above, rss has to be included sobolev space.
{% endhint %}

&#x20;The function minimizing RSS above is the natural cubic spline. The proof is following.

Proof)

Ex5.7 Derivation of smoothing splines (Green and Silverman, 1994). Suppose that $$N \geq 2$$, and that $$g$$ is the natural cubic spline interpolant to the pairs $$\{x_i,z_i\}^N_1$$, with $$a<x_1<\cdots<x_N<b$$. This is a natural spline with a knot at every $$x_i$$; being an N-dimensional space of functions, we can determine the coefficients such that it interpolates the sequence $$z_i$$exactly. Let $$\tilde{g}$$be any other differentiable function on $$[a,b]$$ that interpolates the N pairs.



(a) Let $$h(x)=\tilde{g}(x)-g(x)$$. Use integration by parts and the fact that g is a natural cubic spline to show that

$$
\int^b_a g''(x)h''(x)dx=-\sum^{N-1}_{j=1}g'''(x_j^+)\{h(x_{j+1})-h(x_j)\}=0
$$

(b) Hence show that&#x20;

$$
\int^b_a\tilde{g}''(t)^2dt \geq \int^b_a g''(t)^2dt
$$

and that equality can only hold if h is identically zero in $$[a,b]$$.

(c) Consider the penalized least squares problem&#x20;

$$
min_f[\sum^N_{i=1}(y_i-f(x_i))^2+\lambda\int^b_af''(t)^2dt].
$$

Use (b) to argue that the minimizer must be a cubic spline with knots at each of the $$x_i$$



![](<../../.gitbook/assets/image (34).png>)

> John L. Weatherwax & David Epstein, A Solution Manual and Notes for: The Elements of Statistical Learning by Jerome Friedman, Trevor Hastie, and Robert Tibshirani, 1 March 2021



&#x20;Because this function is the natural cubic spline, we can narrow down the problem from the estimation of function to the estimation of theta.

$$
RSS(\theta,\lambda)=(y-N\theta)^T(y-N\theta)+\lambda\theta^T\Omega_N\theta \\
where \; \{N\}_{ij}=N_j(x_i)  \; and \; \{\Omega_N\}_{jk}=\int N''_j(t)N''_k(t)dt.
$$

$$
\\
\hat{\theta}=(N^TN+\lambda\Omega_N)^{-1}N^Ty \\
\hat{f}(x)=\sum^N_{j=1}N_j(x)\hat{\theta}_j
$$

&#x20;The form of theta is similar with the theta in ridge regression. A ridge regression has an identity matrix instead of omega matrix, so a smoothing spline is a more general expression.&#x20;

