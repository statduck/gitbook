# Statistical Decision Theory

## Statistical Decision Theory

When it comes to decide which model we will use, the important thing to consider is to minimize error. Let's generalize it.

$$
X \in \mathbb{R}^p, Y\in \mathbb{R}, f:\mathbb{R}^p \; to \; \mathbb{R}
$$

The goal is to find the $$f(X)$$ that predict $$Y$$ well. Loss function is needed to find this $$f(X)$$, and this function gives penalizing errors in prediction of $$L(Y, f(X))$$. The square error loss method is a common loss function.

$$
EPE(f)=E(Y-f(X))^2=\int [y-f(x)]^2Pr(dx,dy)=E_XE_{Y|X}([Y-f(X)]^2|X)
$$

<mark style="background-color:yellow;">**✏️ The goal is minimizing the EPE**</mark>

$$
minE_X[g(x)] = min(g(x)) \\
minE_XE_{Y|X}([Y-f(X)]^2|X) =minE_{Y|X}([Y-f(X)]^2|X) \\
f(x) = argmin_cE_{Y|X}([Y-c]^2|X=x)
$$

In terms of c, the solutions is as follow:

$$
f(x)=E(Y|X=x)
$$

**Conditional expectation.** We didn't give any penaltiy such as linearity assumption on f(x). However, optimal f is eventually a conditional expectation in this context.



What is the conditional expectation? To think about this we first deal with a conditional distribution. There are probability assumption on Input variable and Target variable both.



&#x20;🎲**Dice Example** 🎲

&#x20;   Dice is rolled, and let $$X_2$$be the number of possible event of $$4$$ and $$6$$. When we rolled a dice $$n$$ times, if $$4$$ and $$6$$ show up, we successively roll the dice.($$4$$ and $$6$$ doesn't show up -> Stop rolling the dice.)

&#x20;   Let$$X_1$$be the number of possible event of head in coin flip. We flip coin if the result of rolling dice is $$4$$ and $$6$$.&#x20;

$$
p(x_2)=\frac{2}{3}(\frac{1}{3})^{x_2-1}, \quad x_2=1,2,3,... \\
p(x_1|x_2)=\binom {x_2} {x_1}(\frac{1}{2})^{x_2}, \quad x_1=0,1,...,x_2
$$

&#x20;   Our aim is to predict $$X_1$$by using $$X_2$$. When we roll a dice in $$n$$ times, how can we predict $$X_1$$? We can explain it through conditional expectation. We can get $$E(X_1|X_2)$$, and it is determined by the distribution of $$X_1$$. In this case,  $$E(X_1|X_2)=X_2/2$$.   &#x20;

&#x20;   We can say that this explanation below is reasonable. When we roll a dice 2 times, the number of head of coin would be $$1$$. When we roll a dice 4 times the number would be $$2$$. This predictive model is the very **regression model.** The function that has minimum error is eventually regression model.

> **Thus the best prediction of Y at any point X=x is the conditional mean, when best is measured by average squared error.**



&#x20;<mark style="background-color:yellow;">****</mark> <mark style="background-color:yellow;"></mark><mark style="background-color:yellow;">✏️</mark> <mark style="background-color:yellow;"></mark><mark style="background-color:yellow;">**Conditional Expectation and KNN**</mark>

$$
\hat{f}(x)=Ave(y_i|x_i\in N_k(x))
$$

* Expectation is approximated by averaging over sample data;
* Conditioning at a point is relaxed to conditioning on some region "close" to the target point.

With some conditions, we can show that

$$
As \; N,k \xrightarrow {} \infty, \frac{k}{N}\xrightarrow{}0, \quad \hat{f}(x) \xrightarrow {} E(Y|X=x)
$$

&#x20;   When $$N$$ and $$k$$ go to infinity, the estimate of $$f$$ is equal to the conditional expectation. However, when p go bigger the convergence rate decreases and the speed by which $$f$$ goes to conditional expectation is getting slower.&#x20;



&#x20;**** ✏️ **Conditional Expectation and linear regression.**

$$
f(x)=E(Y|X=x)=x^T\beta \\
f(x)=x^T\beta,\quad \hat{\beta}=(XX^T)^{-1}Xy
$$

&#x20;   The above expression has the assumption of constant $$X$$. When we regard $$X$$ as random variable, the optimal solution of $$\beta$$ is as follows:

$$
f(x) \approx x^T\beta \\
\beta=[E(XX^T)]^{-1}E(XY)
$$

&#x20;   This solution can be interpreted as averaging over training data, which same as the beta in least square method. In conclusion, the conditional expectation is derived in KNN and least square as an approximation of averaging. These two approaches have different assumption.

* Least squares assumes $$f(x)$$ is well approximated by a globally linear function.
* k-nearest neighbors assumes $$f(x)$$ is well approximated by a locally constant function.

****

<mark style="background-color:yellow;">**✏️ Several Loss functions**</mark>

&#x20;   When we predict a binary variable $$G$$, we use zero-one loss function.$$L(k,l)$$has an element $$1$$ in wrong prediction, $$0$$ in correct prediction.

$$
L_{kl}=\{I(k\neq l)\} \\
EPE=E[L(G,\hat{G}(X))], \\ EPE=E_X \sum^K_{k=1}L[g_k,\hat{G}(X)] Pr(g_k|X), \quad by\; double \; expectation \\
\hat{G}(x)=argmin \sum^K_{k=1} L(g_k,g)Pr(g_k|X=x) \\
\hat{G}(x)=argmin\sum\{I(g_k\neq g)\}Pr(g_k|X=x)
\\
 \hat{G}(x)=argmin [1-Pr(g|X=x)]
$$

