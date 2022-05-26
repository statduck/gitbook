# Support Vector Machine

## <mark style="background-color:yellow;">Classification Problem</mark>

<mark style="background-color:yellow;"></mark>![](<../../.gitbook/assets/image (93).png>)

$$
y(x)=w^Tx+b
$$

&#x20;This is the line same to decision boundary. We now define the margin as the perpendicular distance between the line and the closet data point. We just want to find the line maximizing the margin.&#x20;

$$
\max_{w,b} \dfrac{2c}{||w||}, \; s.t. \;y_i(w^Tx_i+b)\geq c, \forall i
$$

&#x20;This problem easily changes into the problem because c just changes the scale of w and b.

$$
\max_{w,b} \dfrac{1}{||w||}, \; s.t.\; y_i(w^Tx_i+b) \geq1,\forall i
$$

&#x20;It is equal to a constrained convex quadratic problem.

$$
\min_{w,b} ||w||^2, \; s.t. \; y_i(w^Tx_i+b) \geq 1, \forall i \\ L(w,b,\alpha)=||w||^2+\sum_i\alpha_i(y_i(w^Tx_i+b)-1)
$$

This primal problem can be changed into a dual problem.

$$
\min_{w,b} \dfrac{1}{2}w^Tw, \; s.t. \; 1-y_i(w^Tx_i+b) \leq0, \forall i \\ L(w,b,\alpha)=\dfrac{1}{2}w^Tw+\sum_i\alpha_i(1-y_i(w^Tx_i+b))
$$

By Taking derivative and set to zero,

$$
\dfrac{\partial L}{\partial w}=w-\sum_i\alpha_iy_ix_i=0 \\ \dfrac{\partial L}{\partial b}=\sum_i \alpha_iy_i=0
$$

Using these equations, the problem becomes more easy

$$
L(w,b,\alpha)=\sum_i \alpha_i-\dfrac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j(x_i^Tx_j), \\ s.t. \; \alpha_i \geq 0, \; \sum_i \alpha_i y_i =0
$$

This is a constrained quadratic programming.

#### KKT condition(1)

In the KKT condition, there is the condition $$\alpha_i g_i(w)=0$$.

$$
\alpha_i(1-y_i(w^Tx_i+b))=0
$$

alpha can be zero, or the other thing can be zero. When alpha is nonzero, the training points which have to be on the decision line are called support vectors.

#### KKT condition(2)

$$\dfrac{\partial L}{\partial w}=0$$, so $$w=\sum_i \alpha_iy_ix_i$$

For a new test point z, we need to compute this

$$
w^Tz+b=\sum_{i \in support\; vectors} \alpha_iy_i(x_iz)+b
$$



* b can be derived from $$1-y_i(w^Tx_i+b)=0, \; with \; \alpha_i>0$$



### Problem 1

Data could be not linearly separable. In this case we use this.

$$
(w^Tx+b)y \geq 1-\xi
$$

$$
\min_{w,b,\xi} ||w||^2+C\sum_i \xi_i, s.t. \; y_i(w^Tx_i+b)\geq1-\xi_i, \; \xi_i\geq0,\forall i \\ \min_{w,b} \dfrac{1}{2}w^Tw, \; s.t. \; 1-y_i(w^Tx_i+b)-\xi_i \leq0, \; \xi_i \geq0, \forall i \\ L(w,b,\alpha,\beta)=\dfrac{1}{2}w^Tw+\sum_i C\xi_i+\alpha_i(1-y_i(w^Tx_i+b)-\xi_i)-\beta_i\xi_i
$$

$$
L(w,b,\alpha,\beta)=\sum_i\alpha_i-\dfrac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j(x_i^Tx_j) \\ \max_\alpha \sum_i \alpha_i - \dfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^Tx_j), \\ s.t. \; C-\alpha_i-\beta_i=0, \; \alpha_i \geq 0, \; \beta_i \geq 0, i=1,...,m, \sum_i \alpha_iy_i=0
$$

It is also a constrained quadratic programming.



### Soft margin SVM with primal form

$$
\min_{w,b,\xi} ||w||^2+C \sum_i \xi_i \\ s.t. \; y_i(w^Tx_i+b) \geq 1-\xi_i, \xi_i \geq 0, \forall i
$$

$$\min_{w,b} ||w||^2+C\sum_i max\{0,1-y_i(w^Tx_i+b)\}$$

It is an unconstrained strong convex problem, but not necessarily differentiable. Because it doesn't be differentiable over domain, we use subgradient method instead of original gradient method.



Anyway, this gradient method is used to find the optimal w.

b is solved by this equation: $$1-y_i(w^Tx_i+b)=0$$

$$J(w,b) = ||w||^2+C\sum max\{0,1-y_i(w^Tx_i+b)\}$$

$$
\nabla^{sub}_w J(w,b)=w-C\sum_i \begin{cases} 0, & 1-y_i(w^Tx_i+b) \leq 0 \\y_ix_i, & 1-y_i(w^Tx_i+b)>0 \end{cases}
$$

$$
\nabla^{sub}_b J(w,b)=-C\sum_i\begin{cases} 0, & 1-y_i(w^Tx_i+b) \leq 0 \\ y_i, & 1-y_i(w^Tx_i+b)>0 \end{cases}
$$

$$w^{new}=w^{old}-\eta \nabla^{sub}_{w}J(w,b), \; \; b^{new}=b^{old}-\eta \nabla^{sub}_{b}J(w,b)$$





### Subgradient method

$$
f(x)=|x|
$$

$$g \; is \; a \; subgradient \;of \;f:X\rightarrow R \; at  \; x \in X \\ for \; any \; y \in X: f(y) \geq f(x)+\langle g,y-x\rangle$$

[\[More about subgradient\]](https://convex-optimization-for-all.github.io/contents/chapter07/2021/03/25/07\_01\_subgradient/)



## <mark style="background-color:yellow;">Regression Problem</mark>

$$
f(x)=x^T\beta+\beta_0 \\ H(\beta,\beta_0)=\Sigma^N_{i=1}V(y_i-f(x_i))+\dfrac{\lambda}{2}||\beta||^2
$$

![](<../../.gitbook/assets/image (123).png>)

&#x20;   In regression problem, we want to permit the error size by $$\epsilon$$. Based on the regression line, the point in the range from $$-\epsilon$$ to $$+\epsilon$$ is regarded as the correct point.&#x20;

$$
V_\epsilon(r)=\begin{cases} 0 & if \;|r|<\epsilon, \\ |r|-\epsilon, & otherwise. \end{cases}
$$

$$
V_H(r)=\begin{cases} r^2/2 & if \; |r| \leq c, \\ c|r|-c^2/2, & |r|>c \end{cases}
$$

> SVMs solve binary classification problems by formulating them as convex optimization problems (Vapnik 1998). The optimization problem entails finding the maximum margin separating the hyperplane, while correctly classifying as many training points as possible. SVMs represent this optimal hyperplane with support vectors. - [Reference](https://link.springer.com/chapter/10.1007/978-1-4302-5990-9\_4)



&#x20;   The problem above can be solved by another optimization problem.

![are the distance from the support vector to the point outside.](<../../.gitbook/assets/image (173).png>)

&#x20;   In the view of support vector, this problem is changed into this form.(let $$\beta$$ be equal to $$w$$)

$$
\min_{w}\dfrac{1}{2}||w||^2+C\cdot\Sigma^N_{i=1}(\xi_i^*+\xi_i), \\ s.t. \;\; y_i-w^Tx_i\leq\varepsilon+\xi_i^* \\ w^Tx_i-y_i\leq\varepsilon+\xi_i \\ \xi_i,\xi_i^*\geq0
$$

$$
L(w,\xi,\xi^*,\lambda,\lambda^*,\alpha,\alpha^*)=\dfrac{1}{2}||w||^2+C\cdot\Sigma(\xi_i+\xi_i^*)+\Sigma\alpha_i^*(y_i-w^Tx_i-\varepsilon-\xi_i^*)\\+\Sigma \alpha_i(-y_i+w^Tx_i-\varepsilon-\xi_i)-\Sigma(\lambda_i\xi_i-\lambda_i^*\xi_i^*)
$$

<mark style="color:red;background-color:yellow;">By taking derivative be equal to 0,</mark>

$$
\dfrac{\partial L}{\partial w}=w-\Sigma(\alpha_i^*-\alpha_i)x_i=0 \\ \dfrac{\partial L}{\partial \xi_i^*}=C-\lambda_i^*-\alpha_i^*=0, \;\; \dfrac{\partial L}{\partial \xi_i}=C-\lambda_i-\alpha_i=0 \\ \dfrac{\partial L}{\partial \lambda_i^*}=\Sigma \xi_i^* \leq 0, \;\; \dfrac{\partial L}{\partial \lambda_i}=\Sigma \xi_i \leq 0 \\  \dfrac{\partial L}{\partial \alpha_i^*}=y_i-w^Tx_i-\varepsilon-\xi_i^*\leq 0, \;\; \dfrac{\partial L}{\partial \alpha_i}=-y_i+w^Tx_i-\varepsilon-\xi_i\leq0
$$

<mark style="color:red;background-color:yellow;">Using KKT condition,</mark>

$$
\alpha_i(-y_i+w^Tx_i-\varepsilon-\xi_i)=0\\ \alpha_i^*(y_i-w^Tx_i-\varepsilon-\xi_i^*)=0\\ \lambda_i\xi_i=0, \;\; \lambda_i^*\xi_i^*=0\\ \alpha_i,\alpha_i^*\geq0
$$

The form is summarized like this.

$$
\hat{\beta}=\sum^N_{i=1}(\hat{\alpha}_i^*-\hat{\alpha}_i)x_i, \;\;\hat{f}(x)=\sum^N_{i=1}(\hat{\alpha}_i^*-\hat{\alpha}_i)\langle x,x_i\rangle+\beta_0
$$

$$\hat{\alpha}_i^*,\hat{\alpha}_i$$ are the optimization parameters of this quadratic problem.(It is just a form which replaces some elements in Lagrangian expression.)

$$
\min_{\alpha_i,\alpha_i^*} \varepsilon \sum^N_{i=1}(\alpha^*_i+\alpha_i)-\sum^N_{i=1}y_i(\alpha_i^*-\alpha_i)+\dfrac{1}{2}\sum^N_{i,i'=1}(\alpha_i^*-\alpha_i)(\alpha^*_{i'}-\alpha_{i'})\langle x,x_i \rangle
$$

$$
s.t. \;\;0 \leq \alpha_i,\; \alpha_i^*\leq1/\lambda, \; \sum^N_{i=1}(\alpha_i^*-\alpha_i)=0, \; \alpha_i\alpha_i^*=0
$$

&#x20;   By replacing $$\langle x,x_i\rangle$$with other inner product function, we can generalize this method to richer spaces. $$\varepsilon$$ plays a role of determining how many errors we tolerate.&#x20;



Code From Scratch

```python
# Import library and data
import pandas as pd
import numpy as np
import matplotlib as mpl ; mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt ; plt.rcParams['font.family'] = 'AppleGothic'
import time
from sklearn import datasets

X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10))
y = np.where(y==0, -1, 1)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()

df  = pd.DataFrame(np.column_stack([X,y]), columns=['x1','x2','y'])

def evaluate(w_old, b_old, C):
    idx = np.where(df['y'].values * (df[['x1','x2']].values@w_old+b_old) < 1)[0]
    df_idx = df.iloc[idx]
    
    yixi = (np.expand_dims(df_idx['y'].values,1) * df_idx[['x1','x2']].values)
    yi = df_idx['y'].values

    w_subgrad = w_old-C*sum(yixi)
    b_subgrad = -C*sum(yi)

    return w_subgrad, b_subgrad
    
def batch_subgrad(learning_rate):
    w_old = np.array([0,0]); b_old =0 #initialization
    w_subgrad, b_subgrad = evaluate(w_old, b_old, C=100) 
    diff = 1; i=0
    while(diff > 10e-6):
        w_new = w_old - learning_rate * w_subgrad
        b_new = b_old - learning_rate * b_subgrad
        w_subgrad, b_subgrad = evaluate(w_new, b_new, C=100)
        diff= sum(np.abs(w_new - w_old))
        w_old, b_old = w_new, b_new
        i += 1
        if(i>=20000): break

    print(f'Total iteration: {i}')
    return w_new, b_new

def stoch_subgrad(w_old, b_old, C, learning_rate):
    epoch = 0; diff = 1
    while(diff > 10e-6):
        for x1,x2,y in df.values:
            x = np.array([x1,x2])
            if (y*(x@w_old+b_old) < 1):
                w_subgrad = w_old-C*(y*x)
                b_subgrad = -C*y
            
            else:
                w_subgrad = w_old
                b_subgrad = 0

            w_new = w_old - learning_rate * w_subgrad
            b_new = b_old - learning_rate * b_subgrad
            w_old, b_old = w_new, b_new

        epoch += 1
        if(epoch>=200): break
    print(f'Epochs: {epoch}')
    return w_new, b_new
    
batch_start = time.time()
w1, b1 = batch_subgrad(0.001)
slope1, intercept1 = -w1[0]/w1[1], -b1/w1[1]
batch_end = time.time()

stoch_start = time.time()
w2, b2 = stoch_subgrad(np.array([0,0]), 0, 100, 0.001)
slope2, intercept2 = -w2[0]/w2[1], -b2/w2[1]
stoch_end = time.time()

print('Batch subgradient time: ', batch_end-batch_start)
print('Stochastic subgradient time: ', stoch_end-stoch_start)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter('x1', 'x2', data=df[df['y']==1], color='orange')
ax.scatter('x1', 'x2', data=df[df['y']==-1], color='gray')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Soft Margin SVM')

ax.axline((0, intercept1), slope=slope1, color='black', linewidth=0.8)
ax.axline((0, intercept2), slope=slope2, color='black', linestyle='dashed', linewidth=0.8)
plt.show()
```

![](<../../.gitbook/assets/image (163).png>)

