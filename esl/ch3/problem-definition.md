# Problem Definition

| Cow | Milk(Y) | Age(X1) | Weight(X2) |
| --- | ------- | ------- | ---------- |
| #1  | 10      | 1       | 2          |
| #2  | 11      | 3       | 3          |
| #3  | 12      | 4       | 1          |

&#x20;    We want to find the model which well explains our target variable($$y$$) with $$x$$ variables. The model looks like this&#x20;

$$
Y_i =\beta_1X_{1i}+\beta_2X_{2i}+\epsilon_i
$$

&#x20;   We can evaluate how precise our model it is with a fluctuation of our error. When we assume that our expected error is zero, the fluctuation represents the size of precision.&#x20;

* Good for Intuition: $$E[|\epsilon-E(\epsilon)|]=E[|\epsilon|]$$
* Good for calculation: $$\sqrt{E[\epsilon^2]}=\sigma_\epsilon$$​

&#x20;   If we make a probabilistic assumption for error, we can easily find the fluctuation. For example, Error can be $$-2, -1,0,1,2$$ with the probability $$\dfrac{1}{5}$$. Then $$E[|\epsilon|]=1$$. However, in a real world problem, we couldn't make a probabilistic assumption for error. Even if we do, we just assume the normal with unknown variance. So to know the precision we need to estimate the sigma of error.



MLE: $$\hat{\sigma_\epsilon}=\sqrt{\dfrac{\epsilon_1^2+\cdots+\epsilon_n^2}{n}}$$ | $$\hat{\sigma_\epsilon}=\sqrt{\dfrac{\epsilon_1^2+\cdots\epsilon^2_{n-factor \;num}}{n-factor\;num}}$$

&#x20;



&#x20;
