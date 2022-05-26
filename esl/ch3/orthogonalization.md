# Orthogonalization

## Orthogonalization is important.

### 1) Vector to span{vector}

Let's consider two vectors $$v,w$$in n dimension.

$$
Project \; v \; onto \; span\{w\}=Proj_wv=\dfrac{v \cdot w}{||w||^2}w=\dfrac{v\cdot w}{||w||}\dfrac{w}{||w||}
$$

&#x20;$$\dfrac{v \cdot w}{||w||}$$: length  $$\dfrac{w}{||w||}$$: direction

### Gram-schmidt process

Every non-zero subspace of $$\mathbb{R}^n$$has an orthonormal basis.

1. $$v_1=w_1$$
2. $$v_2=w_2-Proj_{w_1}w_2$$
3. $$v_3=w_3-Proj_{w_2}w_3=w_3-Proj_{v_1,v_2}w_3$$

$$v_k=w_k-\sum^{k-1}_{i=1}Proj_{v_i}w_i$$

$$
\{v_1,v_2,...,v_k\} : \; orthonormal \; basis \; for \; a \; subspace \;W \; of \; \mathbb{R}^n \\
w=(w\cdot v_1)v_1+(w \cdot v_2)v_2+\cdots (w \cdot v_k)v_k
$$

### Projection Matrix

$$A\mathbf{x}=b \;\;\;  would \;\; have \;\; no \;\; solutions. \\ A\hat{\mathbf{x}}=p \;\;\;where \;\; p \; is \; a \; projected \; vector \; from  \; b \; to \; col(A)$$

$$
A\hat{\mathbf{x}}=p \\
A^T(b-A\hat{\mathbf{x}})=0 \\
\hat{\mathbf{x}}=(A^TA)^{-1}A^Tb \\
p=A(A^TA)^{-1}A^Tb=Pb
$$

P$$(A(A^TA)^{-1}A^T)$$ is a symmetric and idempotent matrix.

$$
P=A(A^TA)^{-1}A^T=AA^T=[v_1 \cdots v_n]      \begin{bmatrix}v_1^T\\ \vdots \\v_n^T \end{bmatrix}= v_1\cdot v_1^T + \cdots + v_n \cdot v_n^T \\ when \; A \; has \; an \; orthonormal \; basis, A^TA=I
$$

$$
Pb=(b\cdot v_1)v_1 +\cdots +(b\cdot v_n)v_n
$$



![](../../.gitbook/assets/ch3\_3.jpg)

&#x20;🦊 Regression by Successive Orthogonalization 🦊&#x20;

**Why do we use orthogonalization?**

$$
Y=X\beta +\varepsilon \\
\hat{\beta}=(X^TX)^{-1}X^Ty \\
\hat{\beta}_j=\dfrac{\langle \mathbf{x}_j,\mathbf{y}\rangle}{\langle \mathbf{x}_j,\mathbf{x}_j\rangle}, \quad when \;X=Orthogonal \; matrix \\
$$

&#x20;When X is an orthogonal matrix, simply we can find the coefficient through inner product of j$$_{th}$$input vector and y vector. It is computationally efficient compared to matrix multiplication. However, there is no situation we have orthogonal data in real world. Thus we need to make our data orthogonal.

1. Initialize $$\mathbf{z}_0=\mathbf{x}_0=1$$
2.  For $$j=1,2,\dots,p$$

    $$
    Regress \; \mathbf{x}_j \; on \; \mathbf{z}_0,\mathbf{z}_1,\dots,\mathbf{z}_{j-1} \\
    \mathbf{z}_j=\mathbf{x}_j-\sum^{j-1}_{k=0}\hat{\gamma}_{kj}\mathbf{z}_k, \quad \hat{\gamma}_{lj}=\dfrac{\langle  \mathbf{z}_l,\mathbf{x}_j \rangle}{\langle \mathbf{z}_l, \mathbf{z}_l \rangle} \\
    \mathbf{z}_j=\mathbf{x}_j-\sum^{j-1}_{k=0}\dfrac{\langle  \mathbf{z}_k,\mathbf{x}_j \rangle}{\langle \mathbf{z}_k, \mathbf{z}_k \rangle} \mathbf{z}_k=\mathbf{x}_j-\sum^{j-1}_{k=0} Proj_{\mathbf{z}_k}\mathbf{x}_j
    $$
3. Regress $$\mathbf{y}$$on the residual $$\mathbf{z}_p$$to give the estimate $$\hat{\beta}_p$$

The result of this algorithm is $$\hat{\beta}_p=\dfrac{\langle \mathbf{z}_p,\mathbf{y} \rangle}{\langle \mathbf{z}_p,\mathbf{z}_p \rangle}$$

&#x20; X has an orthonormal basis, so our input vectors all could be expressed as an linear combination of $$z_j$$.&#x20;

$$
x_j=\mathbf{j} \cdot \mathbf{z} \\
$$

We can do extend it to matrix.&#x20;

$$
\begin{split}
\mathbf{X} &=\mathbf{Z}\mathbf{\Gamma}
\\ &= \begin{bmatrix} z_{11} & \cdots & z_{p1} \\ \vdots & \ddots & \vdots \\ z_{n1} & \cdots & z_{np}  \end{bmatrix}
 \begin{bmatrix} \gamma_{11} & \gamma_{12} 
 &\cdots &  \gamma_{1p} \\ 0 & \gamma_{22} & \cdots & \gamma_{2p} \\ 0 & 0 & \cdots & \gamma_{3p} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \gamma_{pp} \end{bmatrix}

\\ &=\mathbf{Z}\mathbf{D}^{-1}\mathbf{D}\mathbf{\Gamma} \\ &=\mathbf{Q}\mathbf{R} \\ 
&= \begin{bmatrix} q_1 & \cdots & q_p \end{bmatrix} \begin{bmatrix} (x_1 \cdot q_1) & (x_2 \cdot q_1) & \cdots & (x_p \cdot q_1)  \\ 0 & (x_2 \cdot q_2) & \cdots & (x_p \cdot q_2) \\ 
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & (x_p \cdot q_p)
\end{bmatrix}

\end{split}
$$

$$\mathbf{Q}$$: direction, $$\mathbf{R}$$: length(scale)

$$
\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}=(\mathbf{R}^T\mathbf{Q}^T\mathbf{Q}\mathbf{R})^{-1}\mathbf{R}^T\mathbf{Q}^Ty=
\mathbf{R}^{-1}\mathbf{Q}^T\mathbf{y} \\
\hat{\mathbf{y}}=\mathbf{Q}\mathbf{Q}^T\mathbf{y}
$$

> It is so computationally efficient!
