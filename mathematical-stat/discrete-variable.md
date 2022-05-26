# Discrete Variable

## Binomial distribution

### Definition

The typical example is binomial distribution. Before we deal with the binomial distribution, we first have to define the bernoulli distribution.

$$
X(success)=1 , \quad X(failure)=0 \\
p(x)=p^x(1-p)^{1-x}, x=0,1
$$

A sequence of bernoulli trial makes binomial distribution.&#x20;

$$E(X)=p, \; Var(X)=p(1-p), \; M(t)=pe^t+q$$

When X is the number of successes in n independent Bernoulli trials, we say that X follows **Binomial distribution**.

### Properties

$$
f(X)=\binom n x p^x(1-p)^{n-x} \\ 
E(X)=np , \; Var(X)=np(1-p) \\
M(t)=(pe^t+q)^n, \; -\infty<t<\infty \\
X \equiv Z_1 +\cdots +Z_n , \; Z_i \sim iid \; Bernoulli(p)
$$

### Example

Let the winning rate of a gambler be 2 over 3. X is the number of winning for 3 games.

$$
p(x)=_3C_x(\dfrac{2}{3})^x(\dfrac{1}{3})^{3-x}, x=0,1,2,3  \\
X \sim B(3,\frac{2}{3})
$$

## Negative Binomial distribution

### Definition

* Number of success : $$r$$
* Number of failures: $$y$$
* The rate of success: $$p$$

&#x20;$$Y$$ is equal to the number of failures until $$r_{th}$$successes.

$$
p(y)=_{y+r-1}C_{r-1} p^{r-1}(1-p)^y*p \\
Y \sim NB(r,p)
$$

###

## Poisson distribution

![](<../.gitbook/assets/image (53).png>)

$$
X \sim Poisson(\lambda), \quad \lambda >0 \\
f(x;\lambda)=\dfrac{\lambda^xe^{-\lambda}}{x!}, \quad x=0,1,2,\cdots \\
P(k \; events \; in \; time \; period)=e^{-\frac{events}{time}*time\;period}\dfrac{(\frac{events}{time}*time \;  period)^k}{k!}
$$

$$\lambda$$is the occurrence number of events per time unit.

> [https://towardsdatascience.com/the-poson-distribution-and-poisson-process-explained-4e2cb17d459](https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459)

![](<../.gitbook/assets/image (59).png>)

### Properties

$$
E(X)=Var(X)=\lambda \\
M(t)=e^{m(e^t-1)} \\
Y = X_1 +\cdots X_n \sim Poisson(\lambda_1+\cdots \lambda_n)
$$

$$
E(X)=\sum^n_{x=0} x\dfrac{\lambda^x e^{-\lambda}}{x!} = \lambda
$$

$$
\begin{split}
M_X(t) &= E(e^{tX}) \\
& = \sum^{\infty}_{x=0} e^{tx} \times \dfrac{\lambda^x e^{-x}}{x!} \\
&= e^{-\lambda} \sum^{\infty}_{x=0}\dfrac{(\lambda e^t)^x}{x!} \\
& = e^{\lambda (e^t-1)}
\end{split}
$$

### Poisson & Binomial

In a rare event situation(n is so large and p is so small in binomial distribution), binomial distribution asymptotically can be a poisson distribution.

$$
\lim_{n \to \infty}\dfrac{n!}{n^x(n-x)!}=1, \; \lim_{n \to \infty}(1-\frac{u}{n})^n=e^{-u}, \; \lim_{n \to \infty}(1-\frac{u}{n})^{-x}=1
$$

$$
\begin{split}
\lim_{n \to \infty}b(x;n,p) &= \lim_{n \to \infty}\dfrac{n!}{x!(n-x)!}p^x(1-p)^{n-x}  \\
&= \lim_{n \to \infty} \dfrac{n^x}{x!} \dfrac{n!}{n^x(n-x)!} p^x(1-p)^{n-x} \\
&= \lim_{n \to \infty} \dfrac{(np)^x(1-p)^{n-x}}{x!} \\
&= \lim_{n \to \infty} \dfrac{(np)^x(1-\frac{np}{n})^{n-x}}{x!} \\
&= \dfrac{\lambda^xe^{-\lambda}}{x!}
\end{split}
$$



### Exercises

![](<../.gitbook/assets/image (62).png>)



![](<../.gitbook/assets/image (63).png>)

