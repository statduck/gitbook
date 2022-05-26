# Smoother Matrices

&#x20;In smoothing splines, the estimated function is like this.

$$
\hat{f}=N(N^TN+\lambda \Omega_N)^{-1}N^Ty=S_\lambda y
$$

In multinomial spline, the estimated function is like this.

$$
\hat{f}=B_\xi(B^T_\xi B_\xi)^{-1}B^T_\xi y=H_\xi y
$$

&#x20;We can compare the matrix $$S_\lambda$$ and $$H_\xi$$

| H(Projection Operator, N\*N) |  S(Smoothing matrix, N\*N)  |
| :--------------------------: | :-------------------------: |
|    Symmetric, $$H \geq 0$$   |   Symmetric, $$S \geq 0$$   |
|     Idempotent ($$HH=H$$)    |      $$SS \leq S$$ (수축)     |
|            Rank M            |            Rank N           |
|        M=trace($$H$$)        | $$df_\lambda$$=trace($$S$$) |

$$
\begin{split}
S_\lambda = \; & N(N^TN+\lambda\Omega_N)^{-1}N^T \\
= \; & (N^{-T}(N^TN+\lambda\Omega_N)N^{-1})^{-1} \\
= \;  & (N^{-T}N^TNN^{-1}+\lambda N^{-T}\Omega_N N^{-1})^{-1} \\
= \; & (I+\lambda N^{-T} \Omega_N N^{-1})^{-1} \\
= \; & (I+\lambda K)^{-1}

\end{split}
$$

$$
min_f(y-f)^T(y-f)+\lambda\theta^T\Omega_N\theta =
min_f(y-f)^T(y-f)+\lambda f^TKf
$$

&#x20;The matrix K is called  penalty matrix.  $$df_\lambda$$ is the effective degree of freedom, which is the number of input variables we use.

$$
S_\lambda=\sum^N_{k=1}\rho_k(\lambda)u_ku_k^T 
 \\
\hat{y}=S_\lambda y= \sum^N_{k=1}\mu_k\rho_k(\lambda)\langle \mu_k,y\rangle \\
\rho_k(\lambda)=\dfrac{1}{1+\lambda d_k}
$$

* $$\hat{y}$$ is decomposed into the sum of N eigen basis vector.
* $$\rho_k(\lambda)$$는 S행렬의 고유값, $$d_k$$ 는 K행렬의 고유값이다. 두 고유값이 역수관계에 놓여있음을 확인할 수 있다.&#x20;



**Smoothing Spline 예시**

![](<../../.gitbook/assets/image (35).png>)

![](<../../.gitbook/assets/image (39).png>)

![](<../../.gitbook/assets/image (38).png>)

붉은 선은 5개의 변수를, 초록 선은 11개의 변수를 $$df_\lambda$$로 잡은 경우이다. 각 고유벡터에 해당하는 고유값은 특정 고유벡터의 크기를 나타내는 것이고 여기서 고유값이 작은 고유벡터는 무시하는 것이다. 우측 하단의 경우에는 각 데이터 포인트에 해당하는 고유벡터의 값을 나타낸 것이다.&#x20;



* 람다에 의해 고유벡터 자체가 바뀌지는 않기 때문에 스무딩 스플라인의 계는 모두 같은 고유벡터를 가진다.
* $$\hat{y}=S_\lambda y=\Sigma^N_{k=1}u_k\rho_k(\lambda)\langle u_k,y \rangle$$의 경우 $$\rho_k(\lambda)$$로 축된 고유벡터들의 선형결합으로 표현된다.
* $$\hat{y}=Hy$$의 경우 고유값이 0또는 1이기 때문에 고유벡터를 살리거나 죽인다는 두 가지 역할만 수행할 수 있다.
* 고유벡터 인덱스 순서는 해당하는 고유값의 크기에 달려있다. 즉 $$u_1$$은 가장 큰 고유값을 가지고 $$u_N$$는 가장 작은 고유값을 가진다.  $$S_\lambda  u_k=\rho_k(\lambda)u_k$$이기 때문에 인덱스가 커질수록(차원이 높아질수록) 스무딩 매트릭스는 고유벡터를 더 축소시킨다고 생각할 수 있다.
* $$u_1,u_2$$의 경우 고유값을 1을 가진다. 이 두가지는 절대 축소되지 않는다. 상수랑 1차로 생각하면 될듯.
* $$\rho_k(\lambda)=1/(1+\lambda d_k)$$
* Reparameterization: $$min _\theta||y-U\theta||^2+\lambda\theta^TD\theta$$, $$U$$는 $$u_k$$를 열로 가지고 $$D$$는 $$d_k$$를 대각성분으로 가지는 대각행렬이다.
* $$df_\lambda=trace(S_\lambda)=\Sigma^N_{k=1}\rho_k(\lambda)$$



