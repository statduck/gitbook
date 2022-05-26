# Calculation

### ✏️ Calculation

![](<../../.gitbook/assets/image (105).png>)

$$
\delta_k(x) =-\dfrac{1}{2}log|\hat{\Sigma}_k|-\dfrac{1}{2}(x-\mu_k)^T\hat{\Sigma}_k^{-1}(x-\mu_k)+log\pi_k
$$

$$\hat{\Sigma}_k=U_kD_kU_k^T$$can make this calculation more faster.

$$
✔️\; (x-\hat{\mu}_k)^T\hat{\Sigma}^{-1}_k(x-\hat{\mu}_k)=[U_k^T(x-\hat{\mu}_k)]^TD_k^{-1}[U_k^T(x-\hat{\mu}_k)] \\
✔️\; log|\hat{\Sigma}_k|=\Sigma_llogd_{kl}
$$

In normal dsitrubiton, the quadratic form means mahalanobis distance. This is the distance that([https://darkpgmr.tistory.com/41](https://darkpgmr.tistory.com/41)) how much each data is away from the mean over the standard deviation.

$$
X^*=D^{-1/2}U^TX  \\
Cov(X^*)=Cov(D^{-1/2}U^TX)=I
$$

Using this expression, we can interpret it as assigning$$X^*$$to the loc $$\mu^*$$. This expression mahalanobis distance can be changed into euclidiean distance. In a transformed space, each point is assigned into the closest median point. It controls the effect of prior probability($$\pi_k$$). The transformed space $$X^*$$is equal to Whitening transformation that makes the variance of X $$\mathbf{I}$$.

$$
Y=WX \\
W^TW=\Sigma^{-1} \\
D^{-1/2}U^T=W
$$

![](<../../.gitbook/assets/image (25).png>)



### ✏️ High Dimensional Data

* $$p$$: The dimension of input matrix
* $$K$$: The number of centroid.

&#x20;   LDA(QDA) reduces $$p$$ dimension into $$K$$. The number of centroids in $$p$$ dimension is $$K$$, so the dimension is reduced into $$K-1$$.  Let $$H_{K-1}$$be a subspace spanned by these centroids. The distance between this subspace and $$X^*$$can be neglected. ( These centroids are already in the subspace, so this distance has same impact on these points. Project transformed $$X^*$$onto $$H_{K-1}$$, and compare distance between the projected points. If the variance of these projected centroids is big, this is an optimal situation. Finding the optimal subspace is same to finding PC space of centroids.&#x20;

* Class centroids $$\mathbf{M} \;(K \times p)$$ , Common covariance matrix $$\mathbf{W}=\hat{\Sigma}=\Sigma^K_{k=1}\Sigma_{g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T/(N-K)$$
* $$\mathbf{M}^*=\mathbf{MW}^{-1/2}$$ using the eigen-decomposition of $$\mathbf{W}$$
* $$\mathbf{B}^*=Cov(\mathbf{M}^*)=\mathbf{V^*D}_B\mathbf{V}^{*T}$$, Columns of $$\mathbf{V}^{*T}$$are $$v_l^*$$
* $$Z_l=v_l^TX \; with \; v_l=\mathbf{W}^{-1/2}v_l^*$$
* $$\mathbf{W}$$: Within-variance matrix, $$\mathbf{B}$$: Between-variance matrix

> Find the linear combination $$Z=a^TX$$such that the between-class variance is maximized relative to the within-class variance.

The problem can be changed into this one:

$$
max_a\dfrac{a^TBa}{a^TWa} \\
max_aa^TBa \quad subject \;\; to \;\; a^TWa=1
$$

The solution $$a$$ is the biggest eigenvector of $$W^{-1}B$$. We can represent our data in a reduced form using the axis of $$Z_l=v_l^TX=a_l^TX$$. $$Z_l$$ is called as a canonical variate and this becomes a new axis.

