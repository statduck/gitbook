# The Effective Number of Parameters

### The Effective Number of Parameters

&#x20;More complex model is, more increased the number of parameter is. Many parameter is good for reducing error, but **it is vulnerable to overfitting.** In this case, we can lower our model complexity by **imposing penalty** on our model. Using this penalty term, we only can choose beta truly necessary in our model. How the number of necessary beta(parameter) is decided? It depends on matrix property multiplied by output vector.&#x20;

&#x20;Degree of freedom is relevant to the number of eigenvalues in Smoother matrix.

> [https://stats.stackexchange.com/questions/114434/what-is-meant-by-effective-parameters-in-machine-learning](https://stats.stackexchange.com/questions/114434/what-is-meant-by-effective-parameters-in-machine-learning)

$$
\hat{y}=Sy \\
df(S)=trace(S)
$$

> Smoother matrix: [https://math.stackexchange.com/questions/2784061/how-to-interpret-the-smoother-matrix](https://math.stackexchange.com/questions/2784061/how-to-interpret-the-smoother-matrix)

&#x20;The degree of freedom in matrix is the number of independent column vectors(or the number of non-zero eigenvalues) We can do eigenvalue decomposition of our smoother matrix because it is symmetric matrix.

$$
S=X(X^TX+\lambda\Omega)^{-1}X^T =UDU^T \\
SS<S (consequence \; of \; shrinkage \; nature)\\
trace(S)=trace(UDU^T)=trace(UU^TD)=trace(D)
$$

&#x20;If it is projection matrix not smoother matrix, trace(D) becomes rank(S) because projection matrix(idempotent matrix) only has 0 or 1 as an eigenvalue. $$SS$$is always small than $$S$$, so eigenvalue is between 0 and 1. I think it would be reason of $$df(S)=trace(S)$$



&#x20;If $$S$$is orthogonal-projection matrix and has M parameters, $$trace(S)=M=df(S)$$. It replaces d in Cp statistic.

$$
Y=f(X)+\epsilon (additive \; model) \\
Var(\epsilon)=\sigma^2_\epsilon
$$

&#x20; In the above assumption, the following expression is satisfied.

$$
\Sigma^n_{i=1}Cov(\hat{y}_i,y_i)=trace(S)\sigma^2_\epsilon \\
df(\hat{y})=\dfrac{\Sigma^n_{i=1} Cov(\hat{y}_i,y_i)}{\sigma^2_\epsilon}
$$

$$
BIC = -2loglik+(logN)d
$$

### VC dimension

The Vapnik-Chervonenkis theory provides such a general measure of complexity, and gives associated bounds on the optimism. It measures the complexity by assessing how wiggly its members can be.

> The VC dimension of the class {f(x,alpha)} is defined to  be the largest number of points that can be shattered by members of {f(x,alpha)}

If our function can perfectly separate the three points into two classes, we can say that the VC dimension of the function is 3(# of points).



> [https://keepmind.net/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-vc-dimension/](https://keepmind.net/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-vc-dimension/)

$$
Err_\mathcal{T}\leq \bar{err}+\frac{\epsilon}{2}(1+\sqrt{1+\frac{4\bar{err}}{\epsilon}}), \quad (binary \; classification) \\
Err_\mathcal{T} \leq \frac{\bar{err}}{(1-c\sqrt{e})_+}, \quad (binary \; classification)
$$













