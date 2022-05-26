# LDA & QDA

## LDA

### ✏️ Goal

The goal is to know the class posterior $$P(G|X)$$.

### ✏️ Expression

$$
P(G=k|X)=\dfrac{P(X|G=k)P(G=k)}{P(X)}=\dfrac{f_k(x)\pi_k}{\Sigma^K_{l=1}f_l(x)\pi_l}
$$

### ✏️ Assumption

The distribution in each class follows multivariate normal distribution with same variance.

$$
(X|G=k) \sim N(\mu_k, \Sigma_k) \\
f_k(x)=\dfrac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}}exp\{-\dfrac{1}{2}(x-\mu_k)^T\Sigma^{-1}_k(x-\mu_k)\}
$$

&#x20;s.t. $$\Sigma_k=\Sigma \forall k$$

### ✏️ Unfolding

Because of the multivariate normal distribution and same variance assumption, log-ratio is easily unfolded.

$$
\begin{split}
log\dfrac{P(G=k|X)}{P(G=l|X)} {} & = log\dfrac{f_k(x)}{f_l(x)}+log\dfrac{\pi_k}{\pi_l} \\
& = log\dfrac{\pi_k}{\pi_l}-\dfrac{1}{2}(\mu_k+\mu_l)^T\Sigma^{-1}(\mu_k-\mu_l) + x^T\Sigma^{-1}(\mu_k-\mu_l)

\end{split}
$$

### ✏️ Classification

Decision boundary is as follows:

$$
\begin{split}
D & =  \{x|P(G=k|X)=P(G=l|X)\} \\
& = \{x|\delta_k(x)=\delta_l(x)\}
\end{split}
$$

$$
P(G=k|X)=P(G=l|X)  \\
☺︎ \\
\dfrac{P(G=k|X)}{P(G=l|X)}=1 \\
☺︎\\
log\dfrac{P(G=k|X)}{P(G=l|X)}=0 \\
☺︎ \\ 
log\dfrac{\pi_k}{\pi_l}-\dfrac{1}{2}(\mu_k+\mu_l)^T\Sigma^{-1}(\mu_k-\mu_l) + x^T\Sigma^{-1}(\mu_k-\mu_l) =0 \\
☺︎ \\
x^T\Sigma^{-1}\mu_k-\dfrac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+log\pi_k = x^T\Sigma^{-1}\mu_l-\dfrac{1}{2}\mu_l^T\Sigma^{-1}\mu_l+log\pi_l \\
☺︎ \\
\delta_k(x)=\delta_l(x)
$$

### ✏️ Parameter estimation

We can't know the parameter of normal distribution in real data, we just estimate this parameter.

$$
\hat{\pi}_k=N_k/N \\
\hat{\mu}_k=\Sigma_{g_i=k}x_i/N_k \\
\hat{\Sigma}=\Sigma^K_{k=1}\Sigma_{g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T/(N-K)
$$

### ✏️ Graph

![](<../../.gitbook/assets/image (79).png>)

![](<../../.gitbook/assets/image (136).png>)

✏️ Another Perspective

| Row  | Height | Weight | Gender |
| ---- | ------ | ------ | ------ |
| Row1 | 5.2    | 1.4    | 0      |
| Row2 | 5.2    | 3.5    | 0      |
| Row3 | 3.5    | 2.2    | 0      |
| Row4 | 3.6    | 5.4    | 1      |
| Row5 | 7.5    | 6.5    | 1      |
| Row6 | 6.6    | 7.5    | 1      |

We can use linear combination of $$a_ 1H+a_2W$$ to make this table into two cluster.



$$
\argmax_{a_1,a_2} E[a_1H+a_2W|G=0]-E[a_2H+a_2W|G=1] \\ s.t. \;Var[a_1H+a_2W]\leq constant
$$

&#x20;It can be more simply changed into this:

$$
\argmax_{a_1,a_2} h(a_1,a_2), \;\; s.t.  \;\;g(a_1,a_2)=c
$$

$$
\mu_1=E[H|G=0]-E[H|G=1] \\ \mu_2=E[W|G=0]-E[W|G=1] \\ h(a_1,a_2)=(a_1 \;\; a_2)(\mu_1 \;\; \mu_2)^T \\ g(a_1,a_2) = a_1^2Var(H|G=0)+2a_1a_2Cov(H,W|G=0)+a_2^2Var(W|G=0)
$$

Using Lagrange multiplier, it is solved.

$$
\nabla g = \lambda \nabla h \\ (a_1 \;\; a_2)=(\mu_1 \;\; \mu_2)\begin{pmatrix}  
 COV(H,H) \; COV(H,W) \\ COV(W,H) \; COV(W,W)\end{pmatrix}^{-1}
$$



## QDA

### ✏️ Assumption

Like LDA, it assumes multivariate normal distribution at each class but there is no same variance assumption.

### ✏️ Classification

$$
D=\{x|\delta_k(x)=\delta_l(x)\} \\\delta_k(x)=-\dfrac{1}{2}log|\Sigma_k|-\dfrac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+log\pi_k
$$

This expression is similar with the expression of LDA, but a quadratic term still remains.

![](<../../.gitbook/assets/image (137).png>)

