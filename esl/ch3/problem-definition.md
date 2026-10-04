# Problem Definition

<table><thead><tr><th>Cow</th><th width="157">Milk(Y)</th><th width="161">Age(X1)</th><th>Weight(X2)</th></tr></thead><tbody><tr><td>#1</td><td>10</td><td>1</td><td>2</td></tr><tr><td>#2</td><td>11</td><td>3</td><td>3</td></tr><tr><td>#3</td><td>12</td><td>4</td><td>1</td></tr></tbody></table>

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
