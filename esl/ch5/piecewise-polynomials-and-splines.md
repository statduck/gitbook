# Piecewise Polynomials and Splines

### ✏️ Local regression using range function.

![](<../../.gitbook/assets/image (29).png>)

$$
f(X)=\beta_1I(X<\xi_1)+\beta_2I(\xi_1\leq X<\xi_2)+\beta_3I(\xi_2 \leq X)
$$

In this case, estimated beta is equal to the mean of target in each area.

![](<../../.gitbook/assets/image (30).png>)

$$
\begin{split}
f(X)= & \beta_1I(X<\xi_1)+\beta_2I(\xi_1\leq X<\xi_2)+\beta_3I(\xi_2 \leq X)+ \\
& \beta_4I(X<\xi_1)X+\beta_5I(\xi_1\leq X<\xi_2)X+\beta_6I(\xi_2\leq X)X \\

& (f(\xi_1^-)=f(\xi_1^+), f(\xi_2^-)=f(\xi_2^+))

\end{split} \\
$$

$$(X-\xi_1)_+$$ can be changed into $$max(0,X-\xi_1)$$.

### ✏️ Piecewise Cubic Polynomials

![](<../../.gitbook/assets/image (31).png>)

$$
f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3+\beta_5(X-\xi_1)^3_++\beta_6(X-\xi_2)^3_+
$$

&#x20; This equation satisfies three constrains that are continuous, first derivative continuous, and second derivative continuous in the border line.$$(X-\xi_k)^3_+$$means this equation satisfies all constrains because it is a cubic function.



**Parameter number**

(# of range) $$\times$$ (# of parameter per range) - (# of knot) $$\times$$(# of constrain per knot) = 3\*4-2\*3=6



&#x20;In lagrange multiplier these two sentences have same meaning,

* Maximize f(x,y), s.t. g(x,y)=k
* Maximize h, s.t. h(x,y,d)=f(x,y)+d(g(x,y)-k)

&#x20;It implies one constraint becomes one term in the lagrange equation. Thus, we minus the number of constrains when we derive the parameter number above.



### ✏️ Weakness of Local polynomial regression

![](<../../.gitbook/assets/image (33).png>)

1. It shows irregular tendency around border lines
2. It's hard to use extrapolation

&#x20;The border lines mean the minimun or maximum of input variables. In these borders the variance of predicted value becomes big.

$$
Point \;wise \;var=Var[\hat{f}(x_0)]
$$

