# Generalized Discriminant Analysis

## <mark style="background-color:yellow;">Generalized LDA</mark>

The strength of LDA is the very **simplicity**.

* It's a simple prototype classifier. One point is classified into the class with closest centroid. Distance is measure with Mahalanobis metric, using a pooled covariance estimate.
* The decision boundary is linear, so it provides simple description and implementation. LDA is informative because it provides low-dimensional view.

The weakness of LDA is the **simplicity** also.

* It is not enough to describe our data just by using two prototypes(class centroid and a common covariance matrix.)
* Linear decision boundary can't adequately separate the data into classes.
* When many features are used, LDA estimates high variance and the performance becomes weak. In this case, we need to restrict or regularize LDA.



## Flexible Discriminant Analysis

<mark style="background-color:yellow;">**Definition**</mark>

This is devised for nonlinear classification.&#x20;

$$
\min_{\beta, \theta}\sum^N_{i=1}(\theta(g_i)-x_i^T\beta)^2
$$

&#x20;   $$g_i$$ is the label for $$i_{th}$$group, and $$\theta$$ is a function mapping with $$G\rightarrow \mathbb{R}^1$$. $$G$$ is a quantitative variable(score) and $$\theta$$ maps this quantitative variable(score) to a categorical value. We call $$\theta(g_i)$$ as transformed class labels which can be predicted by linear regression.

$$
ASR=\dfrac{1}{N}\sum^L_{l=1}[\sum^N_{i=1}(\theta_l(g_i)-x_i^T\beta_l)^2]
$$

$$\theta_l$$ **and** $$\beta_l$$ **are chosen to minimize Average Squared Residual(ASR).**

<mark style="background-color:yellow;">**Matrix Notation**</mark>

* <mark style="background-color:yellow;">****</mark>$$Y$$ is a $$N\times J$$ matrix, $$Y_{ij}=1$$ if $$i_{th}$$ observation falls into $$j_{th}$$ class.
* $$\Theta$$ is a $$J \times K$$ matrix, the column vectors are $$k$$ score vectors for $$j_{th}$$ class.
* $$\Theta^*=Y\Theta$$, it is a transformed class label vector.
* $$ASR(\Theta)=tr(\Theta^{*T}(I-P_X)\Theta^*)/N=tr(\Theta^TY^T(I-P_X)Y\Theta)/N$$, s.t. $$P_X$$ is a projection onto the column space of $$X$$

For reference, $$\sum^N_{i=1}(y_i-\hat{y}_i)^2=y^T(I-P_X)y$$. If the scores($$\Theta^{*T}$$) have mean zero, unit variance, and are uncorrelated for the $$N$$ observation ($$\Theta^{*T}\Theta/N=I_K$$), minimizing $$ASR(\Theta)$$ amounts to finding $$K$$ largest eigenvectors $$\Theta$$ of $$Y^TP_XY$$ with normalization $$\Theta^TD_p\Theta=I_K$$

$$
\min_\Theta tr(\Theta^TY^T(1-P_X)Y\Theta)/N=\min_{\Theta}[tr(\Theta^TY^TY\Theta)/N-tr(\Theta^TY^TP_XY\Theta)/N] \\ =\min_\Theta [K-tr(\Theta^{*T}YP_XY^T\Theta^*)/N]=\max_\Theta tr(\Theta^{*T}S\Theta^*)/N
$$

The theta maximizing trace is the matrix which consists of $$K$$ largest eigenvectors of $$S$$ by Courant-Fischer-characterization of eigenvalues. Finally, we can find an optimal$$\Theta$$.



<mark style="background-color:yellow;">**Implementation**</mark>

1. **Initialize**: Form $$Y$$, $$N\times J$$ indicator matrix.(described above)
2. **Multivariate** **Regression**: Set $$\hat{Y}=P_XY$$, $$\mathbf{B}:\hat{Y}=X\mathbf{B}$$
3. **Optimal** **scores**: Find the eigenvector matrix $$\Theta$$ of $$Y^TY$$=$$Y^TP_XY$$ with normalization $$\Theta^TD_p\Theta=I$$
4. **Update**: $$\mathbf{B}\leftarrow\mathbf{B}\Theta$$

&#x20;   The final regression fit is a$$(J-1)$$ vector function $$\eta(x)=\mathbf{B}^Tx$$. The canonical variates form is as follows.

$$
U^Tx=D\mathbf{B}^Tx=D\eta(x), \;\;s.t. \;D_{kk}^2=\dfrac{1}{\alpha_k^2(1-\alpha_k^2)}
$$

&#x20;    $$\alpha_k^2$$ is the $$k_{th}$$ largest eigenvalue computed in the 3. Optimal scores. We update our coefficient matrix $$\mathbf{B}$$ by using $$\Theta$$which is the eigenvector matrix of $$Y^TP_XY$$. $$U^Tx$$ is the linear canonical variates and $$D\eta(x)$$is a nonparametric version of this discriminant variates. By replacing $$X,P_X$$ with $$h(X),P_{h(x)}=S(\lambda)$$ we can expand it to a nonparametric version. We can call this extended version as a FDA.



<mark style="background-color:yellow;">**Implementation**</mark>

1. **Initialize**: Form $$\Theta_0$$, s.t. $$\Theta^TD_p\Theta=I, \; \Theta_0^*=Y\Theta_0$$
2. **Multivariate** **Nonparametric** **Regression**: Fit $$\hat{\Theta}_0^*=S(\lambda)\Theta_0^*$$, $$\eta(x)=\mathbf{B}^Th(x)$$
3. **Optimal** **scores**: Find the eigenvector matrix $$\Phi$$ of $$\Theta_0^*\hat{\Theta}_0^*=\Theta_0^*S(\hat{\lambda})\Theta_0^{*T}$$. The optimal score is $$\Theta=\Theta_0\Phi$$
4. **Update**: $$\eta(x)=\Phi^T\eta(x)$$

&#x20;   With this implementation, we can get a $$\Phi$$and update $$\eta(x)$$. The final $$\eta(x)$$ is used for calculation of a canonical distance $$\delta(x,j)$$ which is the only thing for classification.

$$
\delta(x,j)=||D(\hat{\eta}(x)-\bar{\eta}(x)^j)||^2
$$



## <mark style="background-color:yellow;">Penalized Discriminant Analysis</mark>

$$
ASR(\{\theta_l,\beta_l\}^L_{l=1}=\dfrac{1}{N}\sum^L_{l=1}[\sum^N_{i=1}(\theta_l(g_i)-h^T(x_i)\beta_l)^2+\lambda\beta_l^T\Omega\beta_l]
$$

When we can choose $$\eta_l(x)=h(x)\beta_l$$, $$\Omega$$ becomes&#x20;

* $$h^T(x_i)=[h_1^T(x_i) \; | \;h_2^T(x_i) \;| \; \cdots \; | \; h_p^T(x_i)]\;$$, we can define $$h_j$$be a vector of up to $$N$$ natural-spline basis function.
* $$S(\lambda)=H(H^TH+\Omega(\lambda))^{-1}H^T$$
* $$ASR_p(\Theta)=tr(\Theta^TY^T(1-S(\lambda))Y\Theta)/N$$
* $$\Sigma_{wthn}+\Omega$$: penalized within-group covariance of $$h(x_i)$$
* $$\Sigma_{btwn}$$: between-group covariance of $$h(x_i)$$
* Find $$argmax_{u} u^T\Sigma_{btwn}u, \;s.t.\;u^T(\Sigma_{wthn}+\Omega)u=1$$, $$u$$ becomes a canonical variate.
* $$D(x,u)=(h(x)-h(u))^T(\Sigma_W+\lambda\Omega)^{-1}(h(x)-h(u))$$,&#x20;



