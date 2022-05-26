# Moment Generating Function

## Expectation



$$
\mu = E(X)=\Sigma xp(x), \int xp(x)dx \\
E(g(X))=\Sigma g(x)p(x), \int g(x)p(x)dx
$$

$$
E(aX)=aE(X) \\
E(X+b)=E(X)+b
$$

$$
\int2f(x)dx=2\int f(x)dx \\
\int (x+2)f(x)dx=\int xf(x)dx+\int 2f(x)dx=\int xf(x)dx+2
$$

$$
E(aX+b)=aE(X)+b \\
E(2X+3)=2E(X)+3, E(X)=5
$$

## Variance

1,2,2,3,3,3,3,4,4,5 -> Small Variance

1,1,2,2,3,3,4,4,5,5 -> Large Variance

$$
\sigma^2=Var(X)=E[(X-\mu)^2] 
\\

E[X^2-2\mu X+\mu^2]=E[X^2]-2\mu E[x]+\mu^2 \\
Var(X)=E[X^2]-\mu^2
$$

$$
f(x)=\dfrac{1}{2}(x+1), \quad -1<x<1
$$

$$
\mu=\int^1_{-1}xf(x)dx=1/2 \int^1_{-1}(x^2+x)dx=1/3 \\
Var(X) = \int^1_{-1} x^2f(x)dx-\mu^2=2/9
$$

### Others

Median: $$P(X \leq 1/2) \; and \; P(X \geq 1/2)$$

Mode: $$argmax_x  \; P(X=x)$$

## Moment generating function

$$
e^{tX}=1+tX+\dfrac{t^2X^2}{2!}+\dfrac{t^3X^3}{3!}+\cdots+\dfrac{t^nX^n}{n!}+\cdots
$$

$$
M(t)=E(e^{tX})=\int e^{tx}f(x)dx \\
M'(t)=\dfrac{d}{dt}E(e^{tX})=\int xe^{tx}f(x)dx \\
M''(t)=\dfrac{d}{d^2t}E(e^{tX})=\int x^2 e^{tx}f(x)dx
$$

$$
M'(t)|_{t=0}=E(X) \\
M''(t)|_{t=0}=E(X^2) \\
\vdots \\
M^{(n)}(t)|_{t=0}=E(X^n)
$$



Moment generating function can determine the distribution. $$E(X^n)$$means $$n_{th}$$moment.



### Uniqueness

Let's assume $$X \sim U(0,1), \quad Y \sim U[0,1]$$

These two random variables make same distribution, but the form of pdf is different. For this reason, pdf can't determine the unique distribution. Only mgf(moment generating function) and cdf(cumulative density function) uniquely define the density function.



### Exercises

$$
X \sim Binom(n,p)
$$

(1) $$E(X)=np$$

$$
\begin{split}
E(X) &= \sum^n_{x=0} x \binom n x p^x(1-p)^{n-x}   \\
&= \sum^n_{x=1} \dfrac{n!}{(x-1)!(n-x)!}p^x(1-p)^{n-x} \\
&= \sum^n_{x=1} np\dfrac{(n-1)!}{(x-1)!(n-x)!}p^{x-1}(1-p)^{n-x} \\
&= np
\end{split}
$$

(2) $$Var(X)=np(1-p)$$

$$
\begin{split}
E(X^2) &= \sum^n_{x=1} xnp\dfrac{(n-1)!}{(x-1)!(n-x)!}p^{x-1}(1-p)^{n-x} \\
&= \sum^n_{x=1} (x-1)np\dfrac{(n-1)!}{(x-1)!(n-x)!}p^{x-1}(1-p)^{n-x} +np \\
&= \sum^n_{x=2} (n-1)np^2\dfrac{(n-2)!}{(x-2)!(n-x)!}p^{x-2}(1-p)^{n-x} +np \\
&= (n-1)np^2 +np
\end{split}
$$

$$
Var(X)=E(X^2)-E(X)^2=(n-1)np^2+np-n^2p^2=np(1-p)
$$



(3) median of X

$$
P(X \leq m) = \sum^m_{x=0}\binom n x p^x(1-p)^{n-x} =1/2
$$

(Case by Case)



(4) mode of X

![](<../.gitbook/assets/image (67).png>)

